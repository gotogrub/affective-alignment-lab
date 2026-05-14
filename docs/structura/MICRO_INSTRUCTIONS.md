# Structura Micro Instructions

These are the build steps distilled from `STRUCTURA_PROJECT_PLAN.md`.

## 1. Contract first

Keep the target contract in `src/structura/schemas.py`.

- `StructuraInput` describes `user_query` plus `retrieved_context`.
- `StructuraOutput` describes intent, slots, selected ids, rejected ids, clarification flags, human handoff and security flags.
- `StructuraSample` is one JSONL row: `id`, `input`, `target`, `scenario`, `split`, `meta`.

Every dataset row and every model prediction must pass this schema before it is trusted.

## 2. Dataset pipeline

Use `data/structura` as the single dataset workspace.

1. Generate catalog:
   `python scripts/structura/generate_catalog.py --output data/structura/raw/catalog_v1.json`
2. Generate JSONL:
   `python scripts/structura/generate_dataset.py --catalog data/structura/raw/catalog_v1.json --output data/structura/processed/structura_smoke.jsonl --num-samples 1000`
3. Validate:
   `python scripts/structura/validate_jsonl.py --path data/structura/processed/train.jsonl --path data/structura/processed/valid.jsonl --path data/structura/processed/test.jsonl`
4. Audit split leakage:
   `python scripts/structura/audit_dataset.py --fail-on-leakage`

The generator must cover product requests, product search, comparisons, no-match cases, ambiguous requests, delivery, returns, order status, technical support, complaints, human handoff and prompt injection.

## 3. Baselines before fine-tuning

Run rules first:

```bash
python scripts/structura/evaluate.py \
  --config configs/structura/eval.yaml \
  --baseline rules
```

This gives a reference for JSON validity, schema validity, product selection and hallucination rate.

## 4. Seq2seq training

Fine-tune FLAN-T5 first because it is small and cheap enough for smoke tests:

```bash
python scripts/structura/train_seq2seq.py \
  --config configs/structura/train_flan_t5_small.yaml
```

Evaluate the checkpoint:

```bash
python scripts/structura/evaluate.py \
  --config configs/structura/eval.yaml \
  --checkpoint outputs/structura/flan-t5-small-smoke \
  --output-predictions data/structura/predictions/flan_t5_small_predictions.jsonl \
  --output-metrics outputs/structura/flan-t5-small-smoke/metrics.json
```

## 5. BERT/RuBERT baseline

Use `scripts/structura/train_bert_baseline.py` for intent classification only. It is not expected to generate JSON; it is a comparison point for understanding and slot extraction work.

## 6. Evaluation gates

For MVP, aim for:

- `valid_json_rate > 0.90`
- `schema_valid_rate > 0.85`
- `intent_accuracy > 0.80`
- `product_selection_f1 > 0.60`
- `hallucination_rate < 0.10`

Never publish synthetic benchmark numbers as final model quality until they are produced by `scripts/structura/evaluate.py`.

## 7. Hugging Face handoff

After training:

1. Write model metrics into `docs/structura/MODEL_CARD_DRAFT.md`.
2. Upload dataset with `scripts/structura/push_to_hub.py --repo-type dataset`.
3. Upload model with `scripts/structura/push_to_hub.py --repo-type model`.
4. Use `app/structura_gradio_demo.py` as the Hugging Face Space entry point.
