from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.dataset import write_json
from structura.synthetic import build_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic Structura product catalog.")
    parser.add_argument("--output", type=Path, default=Path("data/structura/raw/catalog_v1.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-category", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = build_catalog(seed=args.seed, per_category=args.per_category)
    write_json(args.output, catalog)
    print(f"Wrote {len(catalog['products'])} products to {args.output}")


if __name__ == "__main__":
    main()
