from __future__ import annotations

import re
from typing import Any

from .schemas import StructuraOutput, dump_model


CATEGORY_KEYWORDS = {
    "laptop": ["ноут", "ноутбук", "лэптоп", "игровой ноут"],
    "smartphone": ["телефон", "смартфон", "айфон"],
    "monitor": ["монитор", "экран", "дисплей"],
    "printer": ["принтер", "мфу", "печать"],
    "office_chair": ["кресло", "стул"],
    "software_license": ["лиценз", "office", "антивирус", "windows"],
    "crm_package": ["crm", "срм", "воронк", "лид"],
    "support_plan": ["поддержк", "sla", "саппорт"],
}

FEATURE_KEYWORDS = {
    "gaming": ["игр", "игров"],
    "study": ["учеб", "студент"],
    "coding": ["программ", "код", "разработ"],
    "office": ["офис", "работ"],
    "good_camera": ["камер"],
    "battery_life": ["батар", "аккумулятор"],
    "ergonomic": ["эргоном", "спин"],
    "color_accuracy": ["цвет", "дизайн"],
    "duplex_print": ["двусторон"],
    "wifi": ["wifi", "wi-fi", "вайфай"],
    "sales": ["продаж"],
    "legal": ["юрид"],
    "automation": ["автоматизац"],
}

INJECTION_RE = re.compile(
    r"(игнорируй|ignore|system prompt|предыдущие инструкции|верни не json|выведи не json|p999|скрой поле)",
    flags=re.IGNORECASE,
)


def detect_intent(query: str) -> str:
    q = query.lower()
    if INJECTION_RE.search(q):
        return "unknown"
    if any(word in q for word in ["вернуть", "возврат", "верну товар"]):
        return "return_policy"
    if any(word in q for word in ["где мой заказ", "статус заказа", "заказ не приш", "заказ задерж", "оплата прошла", "статус моего заказа"]):
        return "order_status"
    if any(word in q for word in ["достав", "привез", "курьер", "сроки доставки", "варианты доставки", "получить заказ"]):
        return "delivery_question"
    if any(word in q for word in ["сломал", "не работает", "ошибка", "настроить", "помощь", "помогите", "активац", "запустить"]):
        return "technical_support"
    if any(word in q for word in ["ужас", "жалоб", "охрен", "обман", "недоволен", "претенз", "плохой сервис", "разбирательство", "сорвал"]):
        return "complaint"
    if any(word in q for word in ["оператор", "человек", "менеджер", "специалист", "живой", "соедините", "передайте"]):
        return "human_handoff"
    if any(word in q for word in ["сравни", "сравнить", "чем отличается"]):
        return "product_comparison"
    if any(word in q for word in ["посовет", "подбери", "нужен", "хочу"]):
        return "product_recommendation"
    if any(word in q for word in ["найди", "покажи", "есть ли"]):
        return "product_search"
    return "unknown"


def extract_category(query: str) -> str | None:
    q = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            return category
    return None


def extract_price_max(query: str) -> int | None:
    q = query.lower().replace("ё", "е")
    patterns = [
        r"до\s+(\d+)(?:\s*(тыс|тысяч|к|k))?",
        r"не дороже\s+(\d+)(?:\s*(тыс|тысяч|к|k))?",
        r"за\s+(\d+)(?:\s*(тыс|тысяч|к|k))?",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        value = int(match.group(1))
        suffix = match.group(2)
        if suffix or value < 1000:
            value *= 1000
        return value
    return None


def extract_features(query: str) -> list[str]:
    q = query.lower()
    features: list[str] = []
    for feature, keywords in FEATURE_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            features.append(feature)
    return sorted(set(features))


def extract_brand(query: str, context: list[dict[str, Any]]) -> str | None:
    q = query.lower()
    brands = sorted({str(item.get("brand")) for item in context if item.get("brand")})
    for brand in brands:
        if brand.lower() in q:
            return brand
    return None


def extract_constraints(query: str, context: list[dict[str, Any]]) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    price_max = extract_price_max(query)
    brand = extract_brand(query, context)
    features = extract_features(query)

    if price_max is not None:
        constraints["price_max"] = price_max
    if brand is not None:
        constraints["brand"] = brand
    if features:
        constraints["features"] = features
        constraints["use_case"] = [feature for feature in features if feature in {"gaming", "study", "coding", "office", "sales", "legal"}]
    return constraints


def product_score(product: dict[str, Any], category: str | None, constraints: dict[str, Any]) -> int:
    if product.get("type", "product") != "product":
        return -100
    score = 0
    if category:
        if product.get("category") != category:
            return -100
        score += 3
    price_max = constraints.get("price_max")
    if price_max is not None:
        if int(product.get("price", 10**12)) > int(price_max):
            return -100
        score += 2
    brand = constraints.get("brand")
    if brand is not None:
        if product.get("brand") != brand:
            return -100
        score += 2
    requested = set(constraints.get("features") or []) | set(
        constraints.get("use_case") or []
    )
    available = set(product.get("features", [])) | set(product.get("use_cases", []))
    if requested:
        score += len(requested & available)
        if requested.isdisjoint(available):
            score -= 1
    return score


def select_products(context: list[dict[str, Any]], category: str | None, constraints: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []

    for item in context:
        if item.get("type", "product") != "product":
            continue
        score = product_score(item, category, constraints)
        if score >= 3:
            scored.append((score, item))
        elif category is None or item.get("category") == category:
            reason = "does_not_match_constraints" if score < 0 else "weaker_match"
            rejected.append({"id": str(item["id"]), "reason": reason})

    scored.sort(key=lambda pair: (-pair[0], int(pair[1].get("price", 0))))
    selected = [str(item["id"]) for _, item in scored[:2]]
    selected_set = set(selected)
    rejected = [item for item in rejected if item["id"] not in selected_set][:4]
    return selected, rejected


def rules_baseline(user_query: str, retrieved_context: list[dict[str, Any]]) -> dict[str, Any]:
    intent = detect_intent(user_query)
    category = extract_category(user_query)
    constraints = extract_constraints(user_query, retrieved_context)
    security_flags = ["prompt_injection"] if INJECTION_RE.search(user_query) else []
    product_intents = {"product_recommendation", "product_search", "product_comparison"}
    if intent not in product_intents:
        category = None
        constraints = {}
    selected_products: list[str] = []
    rejected_products: list[dict[str, str]] = []
    needs_clarification = False
    clarification_question = None
    needs_human = False
    handoff_reason = None
    answer_type = "unknown"

    if security_flags:
        answer_type = "security_warning"
    elif intent in product_intents:
        selected_products, rejected_products = select_products(retrieved_context, category, constraints)
        if not category and not constraints:
            needs_clarification = True
            clarification_question = "Какой тип товара, бюджет и ключевые требования нужно учесть?"
            answer_type = "ask_clarification"
        elif not selected_products:
            answer_type = "no_match"
        elif intent == "product_comparison":
            answer_type = "compare_products"
        elif intent == "product_search":
            answer_type = "search_products"
        else:
            answer_type = "recommend_products"
    elif intent in {"delivery_question", "return_policy"}:
        answer_type = "answer_policy"
    elif intent == "order_status":
        answer_type = "order_lookup"
        needs_human = True
        handoff_reason = "order_status_requires_customer_data"
    elif intent in {"technical_support", "complaint", "human_handoff"}:
        answer_type = "human_handoff" if intent != "technical_support" else "support_ticket"
        needs_human = True
        handoff_reason = "user_requested_or_requires_operator"
    else:
        needs_clarification = True
        clarification_question = "Уточните, пожалуйста, что именно нужно сделать."
        answer_type = "ask_clarification"

    output = StructuraOutput(
        intent=intent,  # type: ignore[arg-type]
        category=category,
        constraints=constraints,
        selected_products=selected_products,
        rejected_products=rejected_products,
        answer_type=answer_type,  # type: ignore[arg-type]
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        needs_human=needs_human,
        handoff_reason=handoff_reason,
        security_flags=security_flags,
    )
    return dump_model(output)
