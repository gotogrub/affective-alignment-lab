from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.baselines import rules_baseline
from structura.dataset import read_jsonl, write_json, write_jsonl
from structura.formatting import format_prompt
from structura.logging_utils import configure_logging, disk_usage_summary, log_json, log_step, runtime_summary, summarize_lengths
from structura.metrics import evaluate_prediction_records
from structura.validators import validate_output


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
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--debug-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    return parser.parse_args()


def generate_with_checkpoint(
    checkpoint: str,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    debug_samples: int,
    log_every: int,
) -> list[dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    with log_step(logger, f"load checkpoint: {checkpoint}"):
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    generation_config = config.get("generation", {})
    template = config.get("prompt_template", "instruction")
    max_input_length = config.get("data", {}).get("max_input_length", 1024)
    log_json(
        logger,
        "generation_runtime",
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoint": checkpoint,
            "generation_config": generation_config,
            "max_input_length": max_input_length,
            "template": template,
            "records": len(records),
        },
    )

    prediction_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, record in enumerate(records, start=1):
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
        output = prediction_record(record, raw)
        prediction_records.append(output)

        if index <= debug_samples:
            validation = validate_output(raw)
            log_json(
                logger,
                f"generation_sample_{index}",
                {
                    "id": record["id"],
                    "scenario": record.get("scenario"),
                    "prompt_chars": len(prompt),
                    "input_tokens": int(inputs["input_ids"].shape[-1]),
                    "output_tokens": int(output_ids.shape[-1]),
                    "valid_json": validation.valid_json,
                    "schema_valid": validation.schema_valid,
                    "error": validation.error,
                    "raw_output_repr": repr(raw[:800]),
                    "target": record["target"],
                },
            )

        if log_every > 0 and (index % log_every == 0 or index == len(records)):
            elapsed = time.perf_counter() - start
            logger.info(
                "Generated %s/%s predictions in %.2fs (%.2f samples/s)",
                index,
                len(records),
                elapsed,
                index / elapsed if elapsed else 0.0,
            )
    return prediction_records


def collect_validation_errors(records: list[dict[str, Any]], *, limit: int = 25) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for record in records:
        validation = validate_output(record["prediction"])
        if validation.schema_valid:
            continue
        errors.append(
            {
                "id": record.get("id"),
                "scenario": record.get("scenario"),
                "error": validation.error,
                "prediction_preview": str(record.get("prediction", ""))[:1000],
                "target": record.get("target"),
            }
        )
        if len(errors) >= limit:
            break
    return errors


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config) if args.config.exists() else {}
    logging_config = config.get("logging", {})
    logger = configure_logging(args.log_level or logging_config.get("level", "INFO"))
    debug_samples = args.debug_samples if args.debug_samples is not None else int(logging_config.get("debug_samples", 5))
    log_every = args.log_every if args.log_every is not None else int(logging_config.get("log_every", 25))
    logger.info("Structura evaluation started")
    log_json(logger, "runtime", runtime_summary())
    log_json(logger, "config", config)

    default_predictions, default_metrics = default_output_paths(args, config)
    output_predictions = args.output_predictions or default_predictions
    output_metrics = args.output_metrics or default_metrics
    output_errors = output_metrics.parent / "errors.jsonl"
    log_json(
        logger,
        "evaluation_outputs",
        {
            "predictions": str(output_predictions),
            "metrics": str(output_metrics),
            "errors": str(output_errors),
        },
    )

    if args.predictions:
        with log_step(logger, f"load existing predictions: {args.predictions}"):
            prediction_records = read_jsonl(args.predictions)
    else:
        test_path = args.test_path or Path(config.get("data", {}).get("test_path", "data/structura/processed/test.jsonl"))
        with log_step(logger, f"load eval records: {test_path}"):
            records = read_jsonl(test_path)
        if args.limit is not None:
            records = records[: args.limit]
            logger.info("Applied eval limit: %s records", len(records))
        log_json(
            logger,
            "eval_dataset",
            {
                "records": len(records),
                "scenarios": {scenario: sum(1 for record in records if record.get("scenario") == scenario) for scenario in sorted({record.get("scenario") for record in records})},
                "prompt_chars": summarize_lengths([len(format_prompt(record, template=config.get("prompt_template", "instruction"))) for record in records]),
            },
        )

        if args.baseline == "rules" or not args.checkpoint:
            with log_step(logger, "run rules baseline"):
                prediction_records = [
                    prediction_record(
                        record,
                        rules_baseline(record["input"]["user_query"], record["input"].get("retrieved_context", [])),
                    )
                    for record in records
                ]
            for index, record in enumerate(prediction_records[:debug_samples], start=1):
                log_json(
                    logger,
                    f"baseline_sample_{index}",
                    {
                        "id": record["id"],
                        "scenario": record.get("scenario"),
                        "prediction": record["prediction"],
                        "target": record["target"],
                    },
                )
        else:
            prediction_records = generate_with_checkpoint(
                args.checkpoint,
                records,
                config,
                logger=logger,
                debug_samples=debug_samples,
                log_every=log_every,
            )

        with log_step(logger, f"write predictions: {output_predictions}"):
            write_jsonl(output_predictions, prediction_records)
        logger.info("Wrote predictions to %s", output_predictions)

    with log_step(logger, "compute metrics"):
        metrics = evaluate_prediction_records(prediction_records)
    validation_errors = collect_validation_errors(prediction_records, limit=int(logging_config.get("error_sample_limit", 25)))
    if validation_errors:
        log_json(logger, "validation_error_samples", validation_errors, level=logging.WARNING)
        write_jsonl(output_errors, validation_errors)
        logger.warning("Wrote validation error samples to %s", output_errors)

    with log_step(logger, f"write metrics: {output_metrics}"):
        write_json(output_metrics, metrics)
    log_json(logger, "disk_after_evaluation", disk_usage_summary(output_metrics))
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    logger.info("Wrote metrics to %s", output_metrics)
    logger.info("Structura evaluation finished")


if __name__ == "__main__":
    main()
