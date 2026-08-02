from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from structura.formatting import format_prompt, format_target


ENVIRONMENT_LOCK = Path("/opt/structura-environment.lock")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_sha() -> str:
    value = os.environ.get("TRAINING_GIT_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError("TRAINING_GIT_SHA must be an immutable commit SHA")
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def dependency_lock_hash(path: Path = ENVIRONMENT_LOCK) -> str:
    if not path.is_file():
        raise RuntimeError("Worker dependency lock is unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    approved = os.environ.get("DEPENDENCY_LOCK_HASH", "")
    if not re.fullmatch(r"[0-9a-f]{64}", approved) or digest != approved:
        raise RuntimeError("Worker dependency lock differs from approved hash")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Structura QLoRA adapter")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolved_dependency_lock_hash = dependency_lock_hash()
    model_ref = config["base_model"]
    data_config = config["data"]
    train_config = config["training"]
    qlora = config["qlora"]
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref["id"], revision=model_ref["revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render(record: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "user", "content": format_prompt(record, template=data_config["prompt_template"])},
            {"role": "assistant", "content": format_target(record["target"])},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    datasets = load_dataset(
        "json",
        data_files={
            "train": data_config["train_path"],
            "validation": data_config["valid_path"],
        },
    ).map(render, remove_columns=None)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora["quant_type"],
        bnb_4bit_use_double_quant=qlora["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref["id"],
        revision=model_ref["revision"],
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=train_config["gradient_checkpointing"]
    )
    peft_config = LoraConfig(
        r=qlora["rank"],
        lora_alpha=qlora["alpha"],
        lora_dropout=qlora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=qlora["target_modules"],
    )
    max_steps = args.smoke_steps if args.smoke_steps is not None else train_config["max_steps"]
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_config["epochs"],
        max_steps=max_steps,
        learning_rate=train_config["learning_rate"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        gradient_checkpointing=train_config["gradient_checkpointing"],
        logging_steps=train_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_config["eval_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        bf16=True,
        seed=train_config["seed"],
        dataset_text_field="text",
        max_length=data_config["max_length"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "structura-qwen3-4b-qlora")) as run:
        mlflow.log_params(
            {
                "repository_commit": git_sha(),
                "base_model_id": model_ref["id"],
                "base_model_revision": model_ref["revision"],
                "dataset_dvc_hash": os.environ["DATASET_DVC_HASH"],
                "config_hash": canonical_hash(config),
                "dependency_lock_hash": resolved_dependency_lock_hash,
                "seed": train_config["seed"],
            }
        )
        result = trainer.train()
        trainer.save_model(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))
        mlflow.log_metrics({key: float(value) for key, value in result.metrics.items()})
        mlflow.log_artifacts(str(output_dir / "adapter"), artifact_path="adapter")
        manifest = {
            "repository_commit": git_sha(),
            "base_model": model_ref,
            "dataset_dvc_hash": os.environ["DATASET_DVC_HASH"],
            "config": config,
            "config_hash": canonical_hash(config),
            "dependency_lock_hash": resolved_dependency_lock_hash,
            "seed": train_config["seed"],
            "hardware_snapshot": {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "cuda_version": torch.version.cuda,
                "torch_version": torch.__version__,
            },
            "mlflow_run_id": run.info.run_id,
            "metrics": result.metrics,
        }
        (output_dir / "training-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
