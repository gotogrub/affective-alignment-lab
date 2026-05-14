# Affective Alignment Lab

Repository for ML, data science and structured alignment experiments.

## Structura

`Structura` is the current lab track: a compact RAG-to-JSON assistant that converts a user query plus retrieved context into validated business JSON for e-commerce, support and CRM workflows.

```text
user_query + retrieved_context -> validated StructuraOutput JSON
```

The project compares:

- rules baseline
- BERT/RuBERT intent baseline
- seq2seq JSON generator, starting with `google/flan-t5-small`

## Repository Layout

- `src/structura`: schema, dataset formatting, validation, metrics, baseline and inference code.
- `scripts/structura`: CLI pipeline for catalog generation, dataset generation, validation, training, evaluation and Hugging Face upload.
- `configs/structura`: reproducible train/eval configs.
- `data/structura`: synthetic catalog, smoke dataset, splits and prediction artifacts.
- `notebooks/structura`: dataset previews, prompt previews, smoke training, inference and error analysis.
- `docs/structura`: full plan, micro-instructions, dataset spec, evaluation notes, server training guide and model card draft.
- `app/structura_gradio_demo.py`: local or Hugging Face Space demo.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Generate and validate the smoke dataset:

```bash
python scripts/structura/generate_catalog.py \
  --output data/structura/raw/catalog_v1.json

python scripts/structura/generate_dataset.py \
  --catalog data/structura/raw/catalog_v1.json \
  --output data/structura/processed/structura_smoke.jsonl \
  --num-samples 1000

python scripts/structura/validate_jsonl.py \
  --path data/structura/processed/train.jsonl \
  --path data/structura/processed/valid.jsonl \
  --path data/structura/processed/test.jsonl

python scripts/structura/audit_dataset.py --fail-on-leakage
```

Run the rules baseline:

```bash
python scripts/structura/evaluate.py \
  --config configs/structura/eval.yaml \
  --baseline rules
```

Train on a GPU server:

```bash
python scripts/structura/train_seq2seq.py \
  --config configs/structura/train_flan_t5_small.yaml
```

Full server instructions are in `docs/structura/SERVER_TRAINING.md`.
