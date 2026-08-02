from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.dataset import read_jsonl, split_group_key
from structura.formatting import format_prompt, format_target
from structura.validators import validate_sample_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Structura dataset splits for leakage and coverage.")
    parser.add_argument("--train", type=Path, default=Path("data/structura/processed/train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("data/structura/processed/valid.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/structura/processed/test.jsonl"))
    parser.add_argument("--fail-on-leakage", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    p95 = quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
    return {
        "min": min(values),
        "median": median(values),
        "mean": round(mean(values), 2),
        "p95": p95,
        "max": max(values),
    }


def duplicate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[split_group_key(record, group_by="input_target")].append(record)

    duplicate_groups = [group for group in buckets.values() if len(group) > 1]
    cross_split_groups = [group for group in duplicate_groups if len({record["_split"] for record in group}) > 1]
    return {
        "duplicate_input_target_groups": len(duplicate_groups),
        "duplicate_input_target_rows": sum(len(group) for group in duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_groups),
        "cross_split_duplicate_rows": sum(len(group) for group in cross_split_groups),
        "examples": [
            [{"split": record["_split"], "id": record["id"], "scenario": record.get("scenario")} for record in group[:6]]
            for group in cross_split_groups[:5]
        ],
    }


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for split_name, path in {"train": args.train, "valid": args.valid, "test": args.test}.items():
        for record in read_jsonl(path):
            record["_split"] = split_name
            records.append(record)

    invalid = []
    for record in records:
        clean_record = {key: value for key, value in record.items() if not key.startswith("_")}
        errors = validate_sample_record(clean_record)
        if errors:
            invalid.append({"split": record["_split"], "id": record.get("id"), "errors": errors})

    by_split = {}
    for split_name in ("train", "valid", "test"):
        subset = [record for record in records if record["_split"] == split_name]
        by_split[split_name] = {
            "total": len(subset),
            "scenario_counts": dict(sorted(Counter(record.get("scenario", "unknown") for record in subset).items())),
            "intent_counts": dict(sorted(Counter(record["target"]["intent"] for record in subset).items())),
            "category_counts": dict(sorted(Counter(str(record["target"].get("category")) for record in subset).items())),
        }

    prompt_lengths = [len(format_prompt(record)) for record in records]
    target_lengths = [len(format_target(record["target"])) for record in records]

    all_scenarios = {record.get("scenario", "unknown") for record in records}
    coverage_gaps = {
        split_name: sorted(all_scenarios - {record.get("scenario", "unknown") for record in records if record["_split"] == split_name})
        for split_name in ("train", "valid", "test")
    }
    duplicates = duplicate_summary(records)

    summary = {
        "total": len(records),
        "invalid": invalid,
        "splits": by_split,
        "coverage_gaps": coverage_gaps,
        "duplicates": duplicates,
        "prompt_chars": summarize_lengths(prompt_lengths),
        "target_chars": summarize_lengths(target_lengths),
        "selected_count": dict(sorted(Counter(len(record["target"]["selected_products"]) for record in records).items())),
        "needs_clarification": dict(sorted(Counter(record["target"]["needs_clarification"] for record in records).items())),
        "needs_human": dict(sorted(Counter(record["target"]["needs_human"] for record in records).items())),
        "security_flags": dict(sorted(Counter(",".join(record["target"]["security_flags"]) or "none" for record in records).items())),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if invalid or (args.fail_on_leakage and duplicates["cross_split_duplicate_groups"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
