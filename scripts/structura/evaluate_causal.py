from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from structura.constrained_decoding import lm_format_enforcer_schema
from structura.dataset import read_jsonl, write_json, write_jsonl
from structura.formatting import format_prompt
from structura.metrics import evaluate_prediction_records
from structura.schemas import StructuraOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Structura causal model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_model(model_id: str, revision: str, adapter: str | None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.test_path)
    if args.limit:
        records = records[: args.limit]
    model, tokenizer = load_model(args.model, args.revision, args.adapter)
    parser = JsonSchemaParser(
        lm_format_enforcer_schema(StructuraOutput.model_json_schema())
    )
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        tokenizer, parser
    )
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    for record in records:
        prompt = format_prompt(record, template="schema-included")
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        latencies.append((time.perf_counter() - started) * 1000)
        generated = output[0, inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(
            {
                "id": record["id"],
                "input": record["input"],
                "target": record["target"],
                "prediction": raw,
                "scenario": record.get("scenario"),
                "split": record.get("split"),
            }
        )
    metrics = evaluate_prediction_records(predictions)
    ordered = sorted(latencies)
    metrics["latency_ms_p50"] = ordered[len(ordered) // 2] if ordered else 0.0
    metrics["latency_ms_p95"] = (
        ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        if ordered
        else 0.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
