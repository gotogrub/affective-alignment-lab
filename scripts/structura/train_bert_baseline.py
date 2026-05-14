from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.dataset import read_jsonl, write_json
from structura.schemas import Intent


INTENT_LABELS = list(Intent.__args__)  # type: ignore[attr-defined]


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT/RuBERT intent baseline for Structura.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def text_for_record(record: dict[str, Any]) -> str:
    titles = [str(item.get("title", item.get("id", ""))) for item in record["input"].get("retrieved_context", [])]
    return record["input"]["user_query"] + "\n" + " | ".join(titles)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    try:
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training: pip install -r requirements.txt") from exc

    data_config = config["data"]
    training_config = config["training"]
    model_name = config["model_name"]
    label_to_id = {label: index for index, label in enumerate(INTENT_LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    def build_rows(path: str) -> list[dict[str, Any]]:
        rows = []
        for record in read_jsonl(path):
            rows.append({"text": text_for_record(record), "label": label_to_id[record["target"]["intent"]]})
        return rows

    train_dataset = Dataset.from_list(build_rows(data_config["train_path"]))
    valid_dataset = Dataset.from_list(build_rows(data_config["valid_path"]))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=training_config.get("max_length", 256))

    train_dataset = train_dataset.map(tokenize, batched=True)
    valid_dataset = valid_dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(INTENT_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    training_kwargs = {
        "output_dir": training_config["output_dir"],
        "num_train_epochs": training_config.get("num_train_epochs", 3),
        "per_device_train_batch_size": training_config.get("per_device_train_batch_size", 16),
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", 16),
        "learning_rate": training_config.get("learning_rate", 3e-5),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "eval_steps": training_config.get("eval_steps", 200),
        "save_steps": training_config.get("save_steps", 200),
        "logging_steps": training_config.get("logging_steps", 50),
        "report_to": training_config.get("report_to", "none"),
    }
    try:
        training_args = TrainingArguments(evaluation_strategy="steps", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(eval_strategy="steps", **training_kwargs)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])
    write_json(Path(training_config["output_dir"]) / "intent_labels.json", INTENT_LABELS)


if __name__ == "__main__":
    main()
