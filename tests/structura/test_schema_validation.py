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
