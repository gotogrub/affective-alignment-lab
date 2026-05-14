from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import read_jsonl
from .schemas import StructuraOutput, StructuraSample, dump_model, validate_model


@dataclass(frozen=True)
class OutputValidation:
    valid_json: bool
    schema_valid: bool
    parsed: dict[str, Any] | None
    error: str | None = None


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last >= first:
        return stripped[first : last + 1]
    return stripped


def parse_json_object(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(strip_json_fence(value))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def validate_output(value: str | dict[str, Any]) -> OutputValidation:
    try:
        parsed = parse_json_object(value)
    except Exception as exc:
        return OutputValidation(False, False, None, f"invalid_json: {exc}")

    try:
        model = validate_model(StructuraOutput, parsed)
    except Exception as exc:
        return OutputValidation(True, False, parsed, f"schema_error: {exc}")

    return OutputValidation(True, True, dump_model(model), None)


def context_ids(retrieved_context: list[dict[str, Any]]) -> set[str]:
    return {str(item["id"]) for item in retrieved_context if "id" in item}


def product_context_ids(retrieved_context: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["id"])
        for item in retrieved_context
        if "id" in item and item.get("type", "product") == "product"
    }


def hallucinated_product_ids(output: dict[str, Any], retrieved_context: list[dict[str, Any]]) -> list[str]:
    allowed_ids = product_context_ids(retrieved_context)
    return [product_id for product_id in output.get("selected_products", []) if product_id not in allowed_ids]


def validate_sample_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    try:
        sample = validate_model(StructuraSample, record)
    except Exception as exc:
        return [f"schema_error: {exc}"]

    sample_dict = dump_model(sample)
    target = sample_dict["target"]
    allowed_ids = context_ids(sample_dict["input"]["retrieved_context"])
    product_ids = product_context_ids(sample_dict["input"]["retrieved_context"])

    for product_id in target.get("selected_products", []):
        if product_id not in product_ids:
            errors.append(f"selected product id is not grounded in product context: {product_id}")

    for rejected in target.get("rejected_products", []):
        if rejected["id"] not in allowed_ids:
            errors.append(f"rejected product id is not grounded in context: {rejected['id']}")

    if target["needs_clarification"] and not target.get("clarification_question"):
        errors.append("needs_clarification=true requires clarification_question")

    if target["needs_human"] and not target.get("handoff_reason"):
        errors.append("needs_human=true requires handoff_reason")

    return errors


def validate_jsonl_file(path: str | Path) -> dict[str, Any]:
    records = read_jsonl(path)
    invalid: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}

    for index, record in enumerate(records, start=1):
        errors = validate_sample_record(record)
        if errors:
            invalid.append({"line": index, "id": record.get("id"), "errors": errors})
            continue

        scenario = record.get("scenario", "unknown")
        intent = record["target"]["intent"]
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    return {
        "path": str(path),
        "total": len(records),
        "valid": len(records) - len(invalid),
        "invalid": len(invalid),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "errors": invalid,
    }
