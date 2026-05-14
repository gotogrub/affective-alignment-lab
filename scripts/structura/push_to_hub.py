from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Structura artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Example: gotogrub/structura-flan-t5-small")
    parser.add_argument("--path", type=Path, required=True, help="Local model, dataset, or Space directory.")
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    parser.add_argument("--commit-message", default="Upload Structura artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub or requirements.txt before pushing to Hub") from exc

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(args.path),
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.path} to {args.repo_type}:{args.repo_id}")


if __name__ == "__main__":
    main()
