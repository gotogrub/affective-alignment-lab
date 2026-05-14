from __future__ import annotations

import argparse
import inspect
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.dataset import read_jsonl, write_json, write_jsonl
from structura.formatting import format_prompt, to_seq2seq_record
from structura.logging_utils import (
    configure_logging,
    disk_usage_summary,
    log_json,
    log_step,
    runtime_summary,
    summarize_lengths,
)
from structura.validators import validate_output


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for Structura JSON generation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args()


def encode_targets(tokenizer: Any, target_texts: list[str], *, max_target_length: int) -> dict[str, Any]:
    try:
        return tokenizer(text_target=target_texts, max_length=max_target_length, truncation=True)
    except TypeError:
        with tokenizer.as_target_tokenizer():
            return tokenizer(target_texts, max_length=max_target_length, truncation=True)


def count_label_tokens(dataset: Any, pad_token_id: int | None) -> int:
    count = 0
    for row in dataset:
        labels = row["labels"]
        if pad_token_id is None:
            count += len(labels)
        else:
            count += sum(1 for token_id in labels if token_id != pad_token_id)
    return count


def supports_bf16(torch: Any) -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )


def build_training_arguments(
    args_cls: Any,
    training_kwargs: dict[str, Any],
    *,
    eval_strategy: str,
    logger: logging.Logger | None = None,
) -> Any:
    params = inspect.signature(args_cls.__init__).parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    filtered = dict(training_kwargs) if accepts_kwargs else {key: value for key, value in training_kwargs.items() if key in params}

    if "evaluation_strategy" in params or accepts_kwargs:
        filtered["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in params:
        filtered["eval_strategy"] = eval_strategy

    dropped = sorted(set(training_kwargs) - set(filtered))
    if dropped:
        message = f"Skipping unsupported TrainingArguments keys for this transformers version: {dropped}"
        if logger:
            logger.info(message)
        else:
            print(message)

    return args_cls(**filtered)


def build_lightweight_trainer_class(base_trainer_cls: Any) -> Any:
    class LightweightSeq2SeqTrainer(base_trainer_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, save_optimizer_state: bool = True, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.save_optimizer_state = save_optimizer_state

        def _save_optimizer_and_scheduler(self, *args: Any, **kwargs: Any) -> None:
            if self.save_optimizer_state:
                return super()._save_optimizer_and_scheduler(*args, **kwargs)

            output_dir = args[0] if args else kwargs.get("output_dir", "checkpoint")
            print(f"Skipping optimizer/scheduler state save for lightweight checkpoint: {output_dir}")
            return None

    return LightweightSeq2SeqTrainer


def count_by(records: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value: Any = record
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def token_length_summary(dataset: Any, field: str, pad_token_id: int | None = None) -> dict[str, Any]:
    lengths = []
    for row in dataset:
        values = row[field]
        if pad_token_id is None or field != "labels":
            lengths.append(len(values))
        else:
            lengths.append(sum(1 for token_id in values if token_id != pad_token_id))
    return summarize_lengths(lengths)


def post_train_preview(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: str | Path,
    logger: logging.Logger,
) -> None:
    preview_count = int(config.get("logging", {}).get("post_train_preview_samples", 3))
    if preview_count <= 0:
        return

    import torch

    generation_config = config.get("generation", {})
    template = config.get("prompt_template", "instruction")
    max_input_length = config.get("data", {}).get("max_input_length", 1024)
    device = model.device
    model.eval()
    preview_records = []

    with log_step(logger, f"post-train generation preview ({preview_count} samples)"):
        for record in records[:preview_count]:
            prompt = format_prompt(record, template=template)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=generation_config.get("max_new_tokens", 512),
                    num_beams=generation_config.get("num_beams", 1),
                    do_sample=generation_config.get("do_sample", False),
                )
            raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            validation = validate_output(raw)
            item = {
                "id": record["id"],
                "scenario": record.get("scenario"),
                "valid_json": validation.valid_json,
                "schema_valid": validation.schema_valid,
                "error": validation.error,
                "raw_output_preview": raw[:500],
                "target_preview": record["target"],
            }
            preview_records.append(item)
            log_json(logger, "post_train_preview_sample", item)

    path = Path(output_dir) / "post_train_preview.jsonl"
    write_jsonl(path, preview_records)
    logger.info("Wrote post-train preview to %s", path)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    logging_config = config.get("logging", {})
    logger = configure_logging(args.log_level or logging_config.get("level", "INFO"))
    logger.info("Structura seq2seq training started")
    log_json(logger, "runtime", runtime_summary())
    log_json(logger, "config", config)

    try:
        import datasets
        import torch
        import transformers
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training: pip install -r requirements.txt") from exc

    log_json(
        logger,
        "library_versions",
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "bf16_supported": supports_bf16(torch),
        },
    )

    model_name = config["model_name"]
    data_config = config["data"]
    training_config = config["training"]
    template = config.get("prompt_template", "instruction")
    output_dir = Path(training_config["output_dir"])
    log_json(logger, "disk_before_training", disk_usage_summary(output_dir))

    with log_step(logger, f"load tokenizer: {model_name}"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log_json(
            logger,
            "tokenizer",
            {
                "name_or_path": tokenizer.name_or_path,
                "model_max_length": tokenizer.model_max_length,
                "pad_token": tokenizer.pad_token,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token": tokenizer.eos_token,
                "eos_token_id": tokenizer.eos_token_id,
            },
        )

    with log_step(logger, f"load model: {model_name}"):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        log_json(
            logger,
            "model",
            {
                "model_type": getattr(model.config, "model_type", None),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "dtype": str(next(model.parameters()).dtype),
            },
        )

    with log_step(logger, "load jsonl datasets"):
        raw_train_records = read_jsonl(data_config["train_path"])
        raw_valid_records = read_jsonl(data_config["valid_path"])
        train_records = [to_seq2seq_record(record, template=template) for record in raw_train_records]
        valid_records = [to_seq2seq_record(record, template=template) for record in raw_valid_records]
        log_json(
            logger,
            "dataset_counts",
            {
                "train": len(raw_train_records),
                "valid": len(raw_valid_records),
                "train_scenarios": count_by(raw_train_records, ("scenario",)),
                "valid_scenarios": count_by(raw_valid_records, ("scenario",)),
                "train_intents": count_by(raw_train_records, ("target", "intent")),
                "valid_intents": count_by(raw_valid_records, ("target", "intent")),
            },
        )
        log_json(
            logger,
            "text_length_summary",
            {
                "train_prompt_chars": summarize_lengths([len(record["prompt"]) for record in train_records]),
                "train_target_chars": summarize_lengths([len(record["target_text"]) for record in train_records]),
                "valid_prompt_chars": summarize_lengths([len(record["prompt"]) for record in valid_records]),
                "valid_target_chars": summarize_lengths([len(record["target_text"]) for record in valid_records]),
            },
        )
        for index, record in enumerate(train_records[: int(logging_config.get("sample_records", 2))], start=1):
            log_json(
                logger,
                f"train_sample_{index}",
                {
                    "id": record["id"],
                    "prompt_preview": record["prompt"][:700],
                    "target_text": record["target_text"],
                },
            )

    train_dataset = Dataset.from_list(train_records)
    valid_dataset = Dataset.from_list(valid_records)

    max_input_length = training_config.get("max_input_length", 1024)
    max_target_length = training_config.get("max_target_length", 512)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(batch["prompt"], max_length=max_input_length, truncation=True)
        labels = encode_targets(tokenizer, batch["target_text"], max_target_length=max_target_length)
        inputs["labels"] = labels["input_ids"]
        return inputs

    with log_step(logger, "tokenize datasets"):
        tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
        tokenized_valid = valid_dataset.map(tokenize, batched=True, remove_columns=valid_dataset.column_names)

    label_token_count = count_label_tokens(tokenized_train, tokenizer.pad_token_id)
    if label_token_count == 0:
        raise RuntimeError("All target labels are empty after tokenization; refusing to start training.")

    first_train = tokenized_train[0]
    log_json(
        logger,
        "tokenization_preview",
        {
            "train_records": len(train_records),
            "valid_records": len(valid_records),
            "first_input_tokens": len(first_train["input_ids"]),
            "first_target_tokens": len(first_train["labels"]),
            "total_train_label_tokens": label_token_count,
            "target_preview": train_records[0]["target_text"][:180],
        },
    )
    log_json(
        logger,
        "token_length_summary",
        {
            "train_input_tokens": token_length_summary(tokenized_train, "input_ids"),
            "train_target_tokens": token_length_summary(tokenized_train, "labels", tokenizer.pad_token_id),
            "valid_input_tokens": token_length_summary(tokenized_valid, "input_ids"),
            "valid_target_tokens": token_length_summary(tokenized_valid, "labels", tokenizer.pad_token_id),
        },
    )

    num_train_epochs = training_config.get("num_train_epochs", 3)
    train_batch_size = training_config.get("per_device_train_batch_size", 4)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    steps_per_epoch = math.ceil(len(tokenized_train) / max(1, train_batch_size * gradient_accumulation_steps))
    total_train_steps = max(1, int(math.ceil(float(num_train_epochs) * steps_per_epoch)))
    warmup_steps = training_config.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(total_train_steps * training_config.get("warmup_ratio", 0.03))

    fp16_requested = bool(training_config.get("fp16", False) and torch.cuda.is_available())
    force_fp16 = bool(training_config.get("force_fp16", False))
    model_type = getattr(model.config, "model_type", "")
    fp16_enabled = fp16_requested
    if fp16_requested and model_type in {"t5", "mt5"} and not force_fp16:
        print("Disabling fp16 for T5/MT5 stability. Set force_fp16: true only if you know this GPU/model combo is stable.")
        fp16_enabled = False

    bf16_setting = training_config.get("bf16", "auto")
    if bf16_setting == "auto":
        bf16_enabled = supports_bf16(torch) and not fp16_enabled
    else:
        bf16_enabled = bool(bf16_setting and torch.cuda.is_available()) and not fp16_enabled
    if bf16_enabled:
        logger.info("Using bf16 training.")

    training_kwargs = {
        "output_dir": training_config["output_dir"],
        "run_name": config.get("run_name"),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": training_config.get("learning_rate", 5e-5),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "warmup_steps": warmup_steps,
        "fp16": fp16_enabled,
        "bf16": bf16_enabled,
        "logging_steps": training_config.get("logging_steps", 50),
        "eval_steps": training_config.get("eval_steps", 500),
        "save_steps": training_config.get("save_steps", 500),
        "save_total_limit": training_config.get("save_total_limit", 3),
        "overwrite_output_dir": training_config.get("overwrite_output_dir", True),
        "save_only_model": training_config.get("save_only_model", False),
        "save_safetensors": training_config.get("save_safetensors", True),
        "save_strategy": "steps",
        "predict_with_generate": True,
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
        "logging_nan_inf_filter": training_config.get("logging_nan_inf_filter", False),
        "report_to": training_config.get("report_to", "none"),
    }
    log_json(
        logger,
        "training_plan",
        {
            "epochs": num_train_epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_train_steps": total_train_steps,
            "warmup_steps": warmup_steps,
            "effective_batch_size": train_batch_size * gradient_accumulation_steps,
            "fp16": fp16_enabled,
            "bf16": bf16_enabled,
            "save_optimizer_state": training_config.get("save_optimizer_state", True),
        },
    )
    training_args = build_training_arguments(Seq2SeqTrainingArguments, training_kwargs, eval_strategy="steps", logger=logger)
    log_json(
        logger,
        "training_args_effective",
        {
            "output_dir": str(training_args.output_dir),
            "eval_strategy": str(getattr(training_args, "eval_strategy", getattr(training_args, "evaluation_strategy", None))),
            "save_strategy": str(getattr(training_args, "save_strategy", None)),
            "logging_steps": getattr(training_args, "logging_steps", None),
            "eval_steps": getattr(training_args, "eval_steps", None),
            "save_steps": getattr(training_args, "save_steps", None),
            "save_total_limit": getattr(training_args, "save_total_limit", None),
            "save_only_model": getattr(training_args, "save_only_model", None),
            "report_to": getattr(training_args, "report_to", None),
        },
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_valid,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer_cls = build_lightweight_trainer_class(Seq2SeqTrainer)
    with log_step(logger, "initialize trainer"):
        trainer = trainer_cls(**trainer_kwargs, save_optimizer_state=training_config.get("save_optimizer_state", True))

    with log_step(logger, "trainer.train"):
        train_result = trainer.train()
    train_metrics = getattr(train_result, "metrics", {})
    log_json(logger, "train_result_metrics", train_metrics)

    with log_step(logger, "save final model and tokenizer"):
        trainer.save_model(training_config["output_dir"])
        tokenizer.save_pretrained(training_config["output_dir"])
        write_json(Path(training_config["output_dir"]) / "train_metrics.json", train_metrics)

    post_train_preview(
        model=model,
        tokenizer=tokenizer,
        records=raw_valid_records,
        config=config,
        output_dir=training_config["output_dir"],
        logger=logger,
    )
    log_json(logger, "disk_after_training", disk_usage_summary(output_dir))
    logger.info("Structura seq2seq training finished")


if __name__ == "__main__":
    main()
