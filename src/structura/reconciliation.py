from __future__ import annotations

from copy import deepcopy
from typing import Any

from .baselines import extract_category, select_products
from .schemas import StructuraOutput, dump_model, validate_model


PRODUCT_INTENTS = {
    "product_recommendation",
    "product_search",
    "product_comparison",
}
SEARCH_MARKERS = ("покажи", "найди", "есть ли", "список вариантов")
RECOMMENDATION_MARKERS = (
    "посовет",
    "подбери",
    "порекомендуй",
    "что взять",
    "хочу",
    "мне нужен",
    "нужен ",
)


def explicit_product_intent(query: str) -> tuple[str, str] | None:
    """Return only high-confidence intent/category pairs stated in the query."""

    category = extract_category(query)
    if category is None:
        return None
    normalized = query.lower()
    if any(marker in normalized for marker in SEARCH_MARKERS):
        return "product_search", category
    if any(marker in normalized for marker in RECOMMENDATION_MARKERS):
        return "product_recommendation", category
    return None


def reconcile_output(
    output: dict[str, Any],
    *,
    user_query: str,
    retrieved_context: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconcile model semantics with deterministic catalog constraints."""

    reconciled = deepcopy(output)
    explicit = explicit_product_intent(user_query)
    if explicit is not None:
        reconciled["intent"], explicit_category = explicit
        reconciled["category"] = reconciled.get("category") or explicit_category

    intent = reconciled.get("intent")
    if intent in {"product_recommendation", "product_search"}:
        selected, rejected = select_products(
            retrieved_context,
            reconciled.get("category"),
            reconciled.get("constraints") or {},
        )
        reconciled["selected_products"] = selected
        reconciled["rejected_products"] = rejected
        reconciled["answer_type"] = (
            "no_match"
            if not selected
            else "search_products"
            if intent == "product_search"
            else "recommend_products"
        )
        reconciled["needs_clarification"] = False
        reconciled["clarification_question"] = None
        reconciled["needs_human"] = False
        reconciled["handoff_reason"] = None
    elif intent not in PRODUCT_INTENTS:
        reconciled["selected_products"] = []
        reconciled["rejected_products"] = []

    return dump_model(validate_model(StructuraOutput, reconciled))
