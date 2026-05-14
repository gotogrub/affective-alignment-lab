from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.dataset import read_json, split_records, write_jsonl
from structura.synthetic import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Structura JSONL data from a synthetic catalog.")
    parser.add_argument("--catalog", type=Path, default=Path("data/structura/raw/catalog_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/structura/processed/structura_v1.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/structura/processed"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--group-by", choices=["input", "target", "input_target"], default="input_target")
    parser.add_argument("--stratify-key", default="scenario")
    parser.add_argument("--no-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = read_json(args.catalog)
    records = generate_dataset(catalog, num_samples=args.num_samples, seed=args.seed)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")

    if not args.no_splits:
        splits = split_records(
            records,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            group_by=args.group_by,
            stratify_key=args.stratify_key,
        )
        for split_name, split_records_ in splits.items():
            path = args.split_dir / f"{split_name}.jsonl"
            write_jsonl(path, split_records_)
            print(f"Wrote {len(split_records_)} {split_name} records to {path}")


if __name__ == "__main__":
    main()
