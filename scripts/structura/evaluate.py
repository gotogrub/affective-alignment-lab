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


def prediction_record(record: dict[str, Any], prediction: str | dict[str, Any]) -> dict[str, Any]:
    output = {
        "id": record["id"],
        "input": record["input"],
        "target": record["target"],
        "prediction": prediction,
    }
    for key in ("scenario", "split", "meta"):
        if key in record:
            output[key] = record[key]
    return output


def run_id_from_checkpoint(checkpoint: str) -> str:
    return Path(checkpoint.rstrip("/")).name or "checkpoint"


def default_output_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path]:
    if args.checkpoint and args.baseline is None:
        run_id = run_id_from_checkpoint(args.checkpoint)
        return (
            Path(f"data/structura/predictions/{run_id}_predictions.jsonl"),
            Path(f"outputs/structura/{run_id}/metrics.json"),
        )

    if args.baseline == "rules" or not args.checkpoint:
        return (
            Path("data/structura/predictions/rules_baseline_predictions.jsonl"),
            Path("outputs/structura/rules_baseline_metrics.json"),
        )

    outputs = config.get("outputs", {})
    return (
        Path(outputs.get("predictions_path", "data/structura/predictions/predictions.jsonl")),
        Path(outputs.get("metrics_path", "outputs/structura/metrics.json")),
    )


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
    model.eval()
    generation_config = config.get("generation", {})
    template = config.get("prompt_template", "instruction")
    max_input_length = config.get("data", {}).get("max_input_length", 1024)

    prediction_records: list[dict[str, Any]] = []
    for record in records:
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
        prediction_records.append(prediction_record(record, raw))
    return prediction_records


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config) if args.config.exists() else {}

    default_predictions, default_metrics = default_output_paths(args, config)
    output_predictions = args.output_predictions or default_predictions
    output_metrics = args.output_metrics or default_metrics

    if args.predictions:
        prediction_records = read_jsonl(args.predictions)
    else:
        test_path = args.test_path or Path(config.get("data", {}).get("test_path", "data/structura/processed/test.jsonl"))
        records = read_jsonl(test_path)
        if args.limit is not None:
            records = records[: args.limit]

        if args.baseline == "rules" or not args.checkpoint:
            prediction_records = [
                prediction_record(
                    record,
                    rules_baseline(record["input"]["user_query"], record["input"].get("retrieved_context", [])),
                )
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
