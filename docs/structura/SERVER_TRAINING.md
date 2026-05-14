# Server Training Instructions

These commands assume a fresh GPU server with Python 3.10+.

## 1. Clone or update

```bash
git clone https://github.com/gotogrub/affective-alignment-lab.git
cd affective-alignment-lab
```

If the repository already exists:

```bash
cd affective-alignment-lab
git pull origin main
```

## 2. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## 3. Check GPU

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
```

## 4. Rebuild data if needed

The repository includes a smoke dataset. Regenerate it when changing the generator:

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
```

## 5. Train FLAN-T5 small

If you previously produced a run with `loss=0`, `grad_norm=nan` or `eval_loss=nan`, remove that checkpoint first:

```bash
rm -rf outputs/structura/flan-t5-small-smoke
```

```bash
python scripts/structura/train_seq2seq.py \
  --config configs/structura/train_flan_t5_small.yaml
```

## 6. Evaluate

```bash
python scripts/structura/evaluate.py \
  --config configs/structura/eval.yaml \
  --checkpoint outputs/structura/flan-t5-small-smoke \
  --output-predictions data/structura/predictions/flan_t5_small_predictions.jsonl \
  --output-metrics outputs/structura/flan-t5-small-smoke/metrics.json
```

## 7. Push to Hugging Face

Login once:

```bash
hf auth login
```

Upload model:

```bash
python scripts/structura/push_to_hub.py \
  --repo-id gotogrub/structura-flan-t5-small \
  --repo-type model \
  --path outputs/structura/flan-t5-small-smoke \
  --commit-message "Upload Structura FLAN-T5 small"
```

Upload dataset:

```bash
python scripts/structura/push_to_hub.py \
  --repo-id gotogrub/structura-ecommerce-rag-json \
  --repo-type dataset \
  --path data/structura/processed \
  --commit-message "Upload Structura synthetic dataset"
```
