from __future__ import annotations

from typing import Any


_JSON_VALUE_DEFINITION = "ServerMindJsonValue"


def lm_format_enforcer_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt boolean open-object schemas for lm-format-enforcer 0.11.x."""

    replaced_open_object = False

    def normalize(value: Any, *, key: str | None = None) -> Any:
        nonlocal replaced_open_object
        if key == "additionalProperties" and value is True:
            replaced_open_object = True
            return {"$ref": f"#/$defs/{_JSON_VALUE_DEFINITION}"}
        if isinstance(value, dict):
            return {
                child_key: normalize(child_value, key=child_key)
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    normalized = normalize(schema)
    if replaced_open_object:
        definitions = normalized.setdefault("$defs", {})
        if _JSON_VALUE_DEFINITION in definitions:
            raise ValueError("JSON Schema definition name is reserved")
        definitions[_JSON_VALUE_DEFINITION] = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"$ref": f"#/$defs/{_JSON_VALUE_DEFINITION}"},
                },
                {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": f"#/$defs/{_JSON_VALUE_DEFINITION}"
                    },
                },
            ]
        }
    return normalized
