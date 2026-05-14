from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.baselines import rules_baseline
from structura.dataset import read_jsonl, write_json, write_jsonl
from structura.formatting import format_prompt
from structura.metrics import evaluate_prediction_records


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Structura predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/structura/eval.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--output-metrics", type=Path, default=None)
    parser.add_argument("--baseline", choices=["rules"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def generate_with_checkpoint(checkpoint: str, records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    generation_config = config.get("generation", {})
    template = config.get("prompt_template", "instruction")
    max_input_length = config.get("data", {}).get("max_input_length", 1024)

    prediction_records: list[dict[str, Any]] = []
    for record in records:
        prompt = format_prompt(record, template=template)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            num_beams=generation_config.get("num_beams", 1),
            do_sample=generation_config.get("do_sample", False),
        )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction_records.append({"id": record["id"], "input": record["input"], "target": record["target"], "prediction": raw})
    return prediction_records


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config) if args.config.exists() else {}

    output_predictions = args.output_predictions or Path(
        config.get("outputs", {}).get("predictions_path", "data/structura/predictions/predictions.jsonl")
    )
    output_metrics = args.output_metrics or Path(config.get("outputs", {}).get("metrics_path", "outputs/structura/metrics.json"))

    if args.predictions:
        prediction_records = read_jsonl(args.predictions)
    else:
        test_path = args.test_path or Path(config.get("data", {}).get("test_path", "data/structura/processed/test.jsonl"))
        records = read_jsonl(test_path)
        if args.limit is not None:
            records = records[: args.limit]

        if args.baseline == "rules" or not args.checkpoint:
            prediction_records = [
                {
                    "id": record["id"],
                    "input": record["input"],
                    "target": record["target"],
                    "prediction": rules_baseline(record["input"]["user_query"], record["input"].get("retrieved_context", [])),
                }
                for record in records
            ]
        else:
            prediction_records = generate_with_checkpoint(args.checkpoint, records, config)

        write_jsonl(output_predictions, prediction_records)
        print(f"Wrote predictions to {output_predictions}")

    metrics = evaluate_prediction_records(prediction_records)
    write_json(output_metrics, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote metrics to {output_metrics}")


if __name__ == "__main__":
    main()
