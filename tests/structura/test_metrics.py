from structura.metrics import evaluate_prediction_records


def test_evaluate_prediction_records_counts_schema_and_grounding() -> None:
    target = {
        "intent": "product_recommendation",
        "category": "laptop",
        "constraints": {"price_max": 70000},
        "selected_products": ["p0001"],
        "rejected_products": [],
        "answer_type": "recommend_products",
        "needs_clarification": False,
        "clarification_question": None,
        "needs_human": False,
        "handoff_reason": None,
        "security_flags": [],
    }
    records = [
        {
            "id": "sample_1",
            "input": {"user_query": "x", "retrieved_context": [{"id": "p0001", "type": "product"}]},
            "target": target,
            "prediction": target,
        }
    ]

    metrics = evaluate_prediction_records(records)

    assert metrics["valid_json_rate"] == 1.0
    assert metrics["schema_valid_rate"] == 1.0
    assert metrics["product_selection_f1"] == 1.0
    assert metrics["hallucination_rate"] == 0.0
