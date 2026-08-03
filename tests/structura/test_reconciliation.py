from copy import deepcopy

from structura.reconciliation import explicit_product_intent, reconcile_output


def model_output(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "intent": "unknown",
        "category": "laptop",
        "constraints": {"price_max": 70000, "features": ["gaming"]},
        "selected_products": ["invented-id"],
        "rejected_products": [],
        "answer_type": "ask_clarification",
        "needs_clarification": True,
        "clarification_question": "Что именно нужно?",
        "needs_human": False,
        "handoff_reason": None,
        "security_flags": [],
    }
    value.update(overrides)
    return value


def context() -> list[dict[str, object]]:
    return [
        {
            "id": "within-budget",
            "type": "product",
            "category": "laptop",
            "price": 65000,
            "features": ["gaming"],
        },
        {
            "id": "too-expensive",
            "type": "product",
            "category": "laptop",
            "price": 90000,
            "features": ["gaming"],
        },
    ]


def test_explicit_product_intent_requires_category_and_marker() -> None:
    assert explicit_product_intent("Найди ноутбук до 70 тысяч") == (
        "product_search",
        "laptop",
    )
    assert explicit_product_intent("Найди что-нибудь") is None
    assert explicit_product_intent("Ноутбук сломался") is None


def test_reconciliation_applies_catalog_rules_without_mutating_raw_output() -> None:
    raw = model_output()
    snapshot = deepcopy(raw)

    result = reconcile_output(
        raw,
        user_query="Найди ноутбук до 70 тысяч для игр",
        retrieved_context=context(),
    )

    assert raw == snapshot
    assert result["intent"] == "product_search"
    assert result["answer_type"] == "search_products"
    assert result["selected_products"] == ["within-budget"]
    assert result["rejected_products"] == [
        {"id": "too-expensive", "reason": "does_not_match_constraints"}
    ]
    assert result["needs_clarification"] is False
    assert result["clarification_question"] is None


def test_reconciliation_returns_no_match_when_catalog_has_no_valid_product() -> None:
    result = reconcile_output(
        model_output(intent="product_recommendation", constraints={"price_max": 1000}),
        user_query="Посоветуй ноутбук до 1 тысячи",
        retrieved_context=context(),
    )

    assert result["selected_products"] == []
    assert result["answer_type"] == "no_match"


def test_reconciliation_accepts_nullable_catalog_constraints() -> None:
    result = reconcile_output(
        model_output(
            intent="product_search",
            constraints={"price_max": None, "features": None, "use_case": None},
        ),
        user_query="Найди ноутбук",
        retrieved_context=context(),
    )

    assert result["selected_products"] == ["within-budget", "too-expensive"]
    assert result["answer_type"] == "search_products"


def test_reconciliation_clears_products_for_non_product_intent() -> None:
    result = reconcile_output(
        model_output(intent="delivery_question", answer_type="answer_policy"),
        user_query="Когда доставите заказ?",
        retrieved_context=context(),
    )

    assert result["selected_products"] == []
    assert result["rejected_products"] == []
    assert result["intent"] == "delivery_question"
