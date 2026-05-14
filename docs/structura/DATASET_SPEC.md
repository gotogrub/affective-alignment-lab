# Structura Dataset Spec

Structura uses JSONL. Each row is one supervised sample.

```json
{
  "id": "sample_000001",
  "scenario": "product_request",
  "split": "train",
  "input": {
    "user_query": "Посоветуй ноутбук до 70 тысяч для игр",
    "retrieved_context": [
      {
        "id": "p0001",
        "type": "product",
        "title": "Acer Ноутбук 1",
        "category": "laptop",
        "price": 68000
      }
    ]
  },
  "target": {
    "intent": "product_recommendation",
    "category": "laptop",
    "constraints": {
      "price_max": 70000,
      "features": ["gaming"]
    },
    "selected_products": ["p0001"],
    "rejected_products": [],
    "answer_type": "recommend_products",
    "needs_clarification": false,
    "clarification_question": null,
    "needs_human": false,
    "handoff_reason": null,
    "security_flags": []
  },
  "meta": {}
}
```

## Supported Scenarios

- `product_request`
- `product_search`
- `product_comparison`
- `no_match`
- `ambiguous`
- `delivery_question`
- `return_policy`
- `order_status`
- `technical_support`
- `complaint`
- `human_handoff`
- `prompt_injection`

## Grounding Rules

- `selected_products` may contain only product ids that exist in `retrieved_context`.
- `rejected_products[].id` must exist in `retrieved_context`.
- Prompt injection samples must not select invented ids such as `p999`.
- Policy docs can be present in context, but only objects with `type == "product"` can be selected as products.

## Splits

The default generator writes grouped, scenario-stratified splits. Exact duplicate `input + target` groups are kept inside a single split to avoid train/test leakage.

- `data/structura/processed/structura_smoke.jsonl`
- `data/structura/processed/train.jsonl`
- `data/structura/processed/valid.jsonl`
- `data/structura/processed/test.jsonl`

Default split: 80 percent train, 10 percent valid, 10 percent test.

Audit the generated splits with:

```bash
python scripts/structura/audit_dataset.py --fail-on-leakage
```
