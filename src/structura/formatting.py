from __future__ import annotations

import json
from typing import Any

from .schemas import StructuraOutput, dump_model, validate_model


SCHEMA_HINT = {
    "intent": "one of the Structura intent labels",
    "category": "product category or null",
    "constraints": "extracted filters and slots",
    "selected_products": "product ids selected only from retrieved_context",
    "rejected_products": [{"id": "product id", "reason": "short machine-readable reason"}],
    "answer_type": "action type for downstream code",
    "needs_clarification": "boolean",
    "clarification_question": "question string or null",
    "needs_human": "boolean",
    "handoff_reason": "reason string or null",
    "security_flags": "list of flags such as prompt_injection",
}


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def format_target(target: dict[str, Any] | StructuraOutput) -> str:
    if isinstance(target, StructuraOutput):
        payload = dump_model(target)
    else:
        payload = dump_model(validate_model(StructuraOutput, target))
    return compact_json(payload)


def format_prompt(sample_or_input: dict[str, Any], *, template: str = "instruction") -> str:
    if "input" in sample_or_input:
        input_payload = sample_or_input["input"]
    else:
        input_payload = sample_or_input

    user_query = input_payload["user_query"]
    context = input_payload.get("retrieved_context", [])

    if template == "minimal":
        return (
            "Convert to JSON.\n"
            f"User query: {user_query}\n"
            f"Retrieved context: {compact_json(context)}\n"
            "Output JSON:"
        )

    if template == "schema-included":
        return (
            "### Task\n"
            "Convert user query and retrieved RAG context into valid JSON. "
            "Use only product ids present in retrieved_context.\n\n"
            "### JSON schema hint\n"
            f"{pretty_json(SCHEMA_HINT)}\n\n"
            "### User query\n"
            f"{user_query}\n\n"
            "### Retrieved context\n"
            f"{compact_json(context)}\n\n"
            "### Output JSON"
        )

    if template == "few-shot":
        return (
            "### Task\n"
            "Return only valid JSON according to Structura schema.\n\n"
            "### Example\n"
            "User query: Нужен ноутбук до 70000 для учебы\n"
            "Retrieved context: [{\"id\":\"p001\",\"category\":\"laptop\",\"price\":68000,"
            "\"use_cases\":[\"study\"]}]\n"
            "Output JSON: {\"intent\":\"product_recommendation\",\"category\":\"laptop\","
            "\"constraints\":{\"price_max\":70000,\"use_case\":[\"study\"]},"
            "\"selected_products\":[\"p001\"],\"rejected_products\":[],"
            "\"answer_type\":\"recommend_products\",\"needs_clarification\":false,"
            "\"clarification_question\":null,\"needs_human\":false,"
            "\"handoff_reason\":null,\"security_flags\":[]}\n\n"
            "### User query\n"
            f"{user_query}\n\n"
            "### Retrieved context\n"
            f"{compact_json(context)}\n\n"
            "### Output JSON"
        )

    if template != "instruction":
        raise ValueError(f"Unknown prompt template: {template}")

    return (
        "### Task\n"
        "Convert user query and retrieved context into valid JSON according to Structura schema. "
        "Return JSON only. Do not invent product ids.\n\n"
        "### User query\n"
        f"{user_query}\n\n"
        "### Retrieved context\n"
        f"{compact_json(context)}\n\n"
        "### Output JSON"
    )


def to_seq2seq_record(sample: dict[str, Any], *, template: str = "instruction") -> dict[str, str]:
    return {
        "id": sample["id"],
        "prompt": format_prompt(sample, template=template),
        "target_text": format_target(sample["target"]),
    }
