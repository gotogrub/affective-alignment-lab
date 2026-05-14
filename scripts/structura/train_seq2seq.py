from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.dataset import read_jsonl
from structura.formatting import to_seq2seq_record


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a seq2seq model for Structura JSON generation.")
    parser.add_argument("--config", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    try:
        from datasets import Dataset
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training: pip install -r requirements.txt") from exc

    model_name = config["model_name"]
    data_config = config["data"]
    training_config = config["training"]
    template = config.get("prompt_template", "instruction")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    if hasattr(model_config, "tie_word_embeddings"):
        model_config.tie_word_embeddings = False
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=model_config)

    train_records = [to_seq2seq_record(record, template=template) for record in read_jsonl(data_config["train_path"])]
    valid_records = [to_seq2seq_record(record, template=template) for record in read_jsonl(data_config["valid_path"])]
    train_dataset = Dataset.from_list(train_records)
    valid_dataset = Dataset.from_list(valid_records)

    max_input_length = training_config.get("max_input_length", 1024)
    max_target_length = training_config.get("max_target_length", 512)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(batch["prompt"], max_length=max_input_length, truncation=True)
        labels = encode_targets(tokenizer, batch["target_text"], max_target_length=max_target_length)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_valid = valid_dataset.map(tokenize, batched=True, remove_columns=valid_dataset.column_names)
    label_token_count = count_label_tokens(tokenized_train, tokenizer.pad_token_id)
    if label_token_count == 0:
        raise RuntimeError("All target labels are empty after tokenization; refusing to start training.")

    first_train = tokenized_train[0]
    print(
        "Structura tokenization preview:",
        {
            "train_records": len(train_records),
            "valid_records": len(valid_records),
            "first_input_tokens": len(first_train["input_ids"]),
            "first_target_tokens": len(first_train["labels"]),
            "total_train_label_tokens": label_token_count,
            "target_preview": train_records[0]["target_text"][:180],
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
        print("Using bf16 training.")

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
        "save_strategy": "steps",
        "predict_with_generate": True,
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
        "logging_nan_inf_filter": training_config.get("logging_nan_inf_filter", False),
        "report_to": training_config.get("report_to", "none"),
    }
    try:
        training_args = Seq2SeqTrainingArguments(evaluation_strategy="steps", **training_kwargs)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(eval_strategy="steps", **training_kwargs)

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
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


if __name__ == "__main__":
    main()
