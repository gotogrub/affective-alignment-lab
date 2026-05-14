# Structura Evaluation

Evaluation is implemented in `src/structura/metrics.py` and exposed through `scripts/structura/evaluate.py`.

## Metrics

- `valid_json_rate`: prediction can be parsed as a JSON object.
- `schema_valid_rate`: parsed JSON passes `StructuraOutput`.
- `exact_match`: normalized full JSON equals target.
- `intent_accuracy`: exact match on `intent`.
- `category_accuracy`: exact match on `category`.
- `answer_type_accuracy`: exact match on `answer_type`.
- `product_selection_precision`, `product_selection_recall`, `product_selection_f1`: set comparison for `selected_products`.
- `product_selection_f1_on_positive`: product F1 only for examples where the target selects at least one product.
- `empty_selection_rate`: fraction of schema-valid predictions with no selected products.
- `empty_selection_on_positive_rate`: fraction of positive-selection targets where the model selected nothing.
- `hallucination_rate`: selected product ids absent from product context divided by predicted selected ids.
- `hallucinated_prediction_rate`: fraction of schema-valid predictions containing at least one hallucinated product id.
- `clarification_f1`: F1 for `needs_clarification`.
- `needs_human_f1`: F1 for `needs_human`.
- `injection_detection_f1`: F1 for `prompt_injection` in `security_flags`.

## Baseline

Run the rules baseline before model training:

```bash
python scripts/structura/evaluate.py \
  --config configs/structura/eval.yaml \
  --baseline rules
```

This writes:

- `data/structura/predictions/rules_baseline_predictions.jsonl`
- `outputs/structura/rules_baseline_metrics.json`

The metrics file also contains `scenario_metrics` for per-scenario debugging.

## Model Evaluation

```bash
python scripts/structura/evaluate.py \
  --config configs/structura/eval.yaml \
  --checkpoint outputs/structura/flan-t5-small-smoke \
  --output-predictions data/structura/predictions/flan_t5_small_predictions.jsonl \
  --output-metrics outputs/structura/flan-t5-small-smoke/metrics.json
```

## Error Analysis

Use `notebooks/structura/05_error_analysis.ipynb` after evaluation. Group errors by:

- invalid JSON
- schema errors
- wrong intent
- wrong category
- missing slot
- wrong product selection
- hallucinated product id
- bad clarification
- bad handoff
- missed prompt injection
