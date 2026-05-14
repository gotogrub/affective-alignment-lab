from __future__ import annotations

from typing import Any

from .formatting import compact_json
from .validators import hallucinated_product_ids, validate_output


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def prf(predicted: set[str], gold: set[str]) -> dict[str, float]:
    true_positive = len(predicted & gold)
    precision = safe_div(true_positive, len(predicted))
    recall = safe_div(true_positive, len(gold))
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def bool_prf(predicted_values: list[bool], gold_values: list[bool]) -> dict[str, float]:
    predicted_positive = {str(index) for index, value in enumerate(predicted_values) if value}
    gold_positive = {str(index) for index, value in enumerate(gold_values) if value}
    return prf(predicted_positive, gold_positive)


def field_accuracy(predictions: list[dict[str, Any] | None], targets: list[dict[str, Any]], field: str) -> float:
    correct = 0
    for prediction, target in zip(predictions, targets):
        if prediction is not None and prediction.get(field) == target.get(field):
            correct += 1
    return safe_div(correct, len(targets))


def exact_match(prediction: dict[str, Any] | None, target: dict[str, Any]) -> bool:
    if prediction is None:
        return False
    return compact_json(prediction) == compact_json(target)


def evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total": 0}

    parsed_predictions: list[dict[str, Any] | None] = []
    targets: list[dict[str, Any]] = []
    valid_json = 0
    schema_valid = 0
    exact = 0
    product_f1_values: list[float] = []
    product_precision_values: list[float] = []
    product_recall_values: list[float] = []
    hallucinated_ids = 0
    predicted_ids = 0
    prediction_errors: dict[str, int] = {}

    for record in records:
        prediction_value = record["prediction"]
        target = record["target"]
        validation = validate_output(prediction_value)
        targets.append(target)

        if validation.valid_json:
            valid_json += 1
        if validation.schema_valid and validation.parsed is not None:
            schema_valid += 1
            prediction = validation.parsed
            parsed_predictions.append(prediction)
            exact += int(exact_match(prediction, target))
            selected_pred = set(prediction.get("selected_products", []))
            selected_gold = set(target.get("selected_products", []))
            product_scores = prf(selected_pred, selected_gold)
            product_precision_values.append(product_scores["precision"])
            product_recall_values.append(product_scores["recall"])
            product_f1_values.append(product_scores["f1"])
            hallucinated = hallucinated_product_ids(prediction, record["input"].get("retrieved_context", []))
            hallucinated_ids += len(hallucinated)
            predicted_ids += len(selected_pred)
        else:
            parsed_predictions.append(None)
            product_precision_values.append(0.0)
            product_recall_values.append(0.0)
            product_f1_values.append(0.0)
            key = (validation.error or "unknown_error").split(":", 1)[0]
            prediction_errors[key] = prediction_errors.get(key, 0) + 1

    clarification_scores = bool_prf(
        [bool(pred and pred.get("needs_clarification")) for pred in parsed_predictions],
        [bool(target.get("needs_clarification")) for target in targets],
    )
    human_scores = bool_prf(
        [bool(pred and pred.get("needs_human")) for pred in parsed_predictions],
        [bool(target.get("needs_human")) for target in targets],
    )
    injection_scores = bool_prf(
        [bool(pred and "prompt_injection" in pred.get("security_flags", [])) for pred in parsed_predictions],
        [bool("prompt_injection" in target.get("security_flags", [])) for target in targets],
    )

    return {
        "total": total,
        "valid_json_rate": safe_div(valid_json, total),
        "schema_valid_rate": safe_div(schema_valid, total),
        "exact_match": safe_div(exact, total),
        "intent_accuracy": field_accuracy(parsed_predictions, targets, "intent"),
        "category_accuracy": field_accuracy(parsed_predictions, targets, "category"),
        "answer_type_accuracy": field_accuracy(parsed_predictions, targets, "answer_type"),
        "needs_clarification_accuracy": field_accuracy(parsed_predictions, targets, "needs_clarification"),
        "needs_human_accuracy": field_accuracy(parsed_predictions, targets, "needs_human"),
        "product_selection_precision": safe_div(sum(product_precision_values), total),
        "product_selection_recall": safe_div(sum(product_recall_values), total),
        "product_selection_f1": safe_div(sum(product_f1_values), total),
        "hallucination_rate": safe_div(hallucinated_ids, predicted_ids),
        "clarification_f1": clarification_scores["f1"],
        "needs_human_f1": human_scores["f1"],
        "injection_detection_f1": injection_scores["f1"],
        "prediction_errors": dict(sorted(prediction_errors.items())),
    }
