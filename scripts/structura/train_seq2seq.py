from __future__ import annotations

import argparse
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


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    try:
        from datasets import Dataset
        import torch
        from transformers import (
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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_records = [to_seq2seq_record(record, template=template) for record in read_jsonl(data_config["train_path"])]
    valid_records = [to_seq2seq_record(record, template=template) for record in read_jsonl(data_config["valid_path"])]
    train_dataset = Dataset.from_list(train_records)
    valid_dataset = Dataset.from_list(valid_records)

    max_input_length = training_config.get("max_input_length", 1024)
    max_target_length = training_config.get("max_target_length", 512)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(batch["prompt"], max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=batch["target_text"], max_length=max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    tokenized_valid = valid_dataset.map(tokenize, batched=True, remove_columns=valid_dataset.column_names)

    training_kwargs = {
        "output_dir": training_config["output_dir"],
        "run_name": config.get("run_name"),
        "num_train_epochs": training_config.get("num_train_epochs", 3),
        "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 4),
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 1),
        "learning_rate": training_config.get("learning_rate", 5e-5),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "warmup_ratio": training_config.get("warmup_ratio", 0.03),
        "fp16": bool(training_config.get("fp16", False) and torch.cuda.is_available()),
        "logging_steps": training_config.get("logging_steps", 50),
        "eval_steps": training_config.get("eval_steps", 500),
        "save_steps": training_config.get("save_steps", 500),
        "save_total_limit": training_config.get("save_total_limit", 3),
        "save_strategy": "steps",
        "predict_with_generate": True,
        "report_to": training_config.get("report_to", "none"),
    }
    try:
        training_args = Seq2SeqTrainingArguments(evaluation_strategy="steps", **training_kwargs)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(eval_strategy="steps", **training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


if __name__ == "__main__":
    main()
