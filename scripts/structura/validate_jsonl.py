from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.validators import validate_jsonl_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Structura JSONL records.")
    parser.add_argument("--path", type=Path, action="append", required=True, help="JSONL file. Can be passed multiple times.")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_errors = False
    for path in args.path:
        summary = validate_jsonl_file(path)
        errors = summary.pop("errors")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        if errors:
            has_errors = True
            print(json.dumps(errors[: args.max_errors], ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
    if has_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
