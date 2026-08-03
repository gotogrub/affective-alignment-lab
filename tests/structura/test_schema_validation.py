from lmformatenforcer import JsonSchemaParser

from structura.constrained_decoding import lm_format_enforcer_schema
from structura.schemas import StructuraOutput, dump_model, validate_model


def test_structura_output_schema_accepts_minimal_valid_output() -> None:
    output = validate_model(
        StructuraOutput,
        {
            "intent": "product_recommendation",
            "category": "laptop",
            "constraints": {"price_max": 70000},
            "selected_products": ["p0001"],
            "rejected_products": [{"id": "p0002", "reason": "over_budget"}],
            "answer_type": "recommend_products",
            "needs_clarification": False,
            "clarification_question": None,
            "needs_human": False,
            "handoff_reason": None,
            "security_flags": [],
        },
    )

    assert dump_model(output)["selected_products"] == ["p0001"]


def test_lm_format_enforcer_accepts_free_form_constraint_values() -> None:
    source_schema = StructuraOutput.model_json_schema()
    compatible_schema = lm_format_enforcer_schema(source_schema)

    assert source_schema["properties"]["constraints"]["additionalProperties"] is True
    assert compatible_schema["properties"]["constraints"]["additionalProperties"] == {
        "$ref": "#/$defs/ServerMindJsonValue"
    }
    assert compatible_schema["additionalProperties"] is False

    parser = JsonSchemaParser(compatible_schema)
    payload = (
        '{"intent":"product_search","constraints":'
        '{"price_max":70000,"brands":["acer"],"flags":{"refurbished":false}}}'
    )
    for character in payload:
        parser = parser.add_character(character)

    assert parser.can_end()
