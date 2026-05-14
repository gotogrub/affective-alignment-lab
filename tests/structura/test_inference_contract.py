from structura.baselines import rules_baseline
from structura.validators import validate_output


def test_rules_baseline_returns_valid_schema_output() -> None:
    context = [
        {
            "id": "p0001",
            "type": "product",
            "title": "Acer Nitro",
            "category": "laptop",
            "price": 68000,
            "features": ["gaming"],
            "use_cases": ["gaming", "study"],
        }
    ]

    prediction = rules_baseline("Нужен ноутбук до 70 тысяч для игр", context)
    validation = validate_output(prediction)

    assert validation.schema_valid
    assert validation.parsed is not None
    assert validation.parsed["selected_products"] == ["p0001"]
