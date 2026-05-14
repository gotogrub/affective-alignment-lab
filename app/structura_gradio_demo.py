from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structura.inference import load_generator
from structura.validators import hallucinated_product_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Structura Gradio demo.")
    parser.add_argument("--checkpoint", default=None, help="Fine-tuned checkpoint path. If omitted, rules baseline is used.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def parse_context(text: str) -> list[dict[str, Any]]:
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("Retrieved context must be a JSON array")
    return value


def main() -> None:
    args = parse_args()
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install gradio or requirements.txt before running the demo") from exc

    generator = load_generator(args.checkpoint)

    default_context = json.dumps(
        [
            {
                "id": "p001",
                "type": "product",
                "title": "Acer Nitro 5",
                "category": "laptop",
                "price": 68000,
                "features": ["gaming"],
                "use_cases": ["gaming", "study"],
            },
            {
                "id": "p002",
                "type": "product",
                "title": "Lenovo IdeaPad 3",
                "category": "laptop",
                "price": 52000,
                "features": ["office"],
                "use_cases": ["study", "office"],
            },
        ],
        ensure_ascii=False,
        indent=2,
    )

    def run(user_query: str, context_text: str) -> tuple[str, str, str]:
        try:
            context = parse_context(context_text)
            result = generator.generate(user_query, context)
            validation = result["validation"]
            parsed = validation.parsed or {}
            hallucinated = hallucinated_product_ids(parsed, context) if parsed else []
            status = {
                "valid_json": validation.valid_json,
                "schema_valid": validation.schema_valid,
                "hallucinated_product_ids": hallucinated,
                "error": validation.error,
            }
            return (
                str(result["raw_output"]),
                json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True),
                json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
            )
        except Exception as exc:
            return "", "", json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)

    with gr.Blocks(title="Structura RAG-to-JSON") as demo:
        gr.Markdown("# Structura RAG-to-JSON")
        with gr.Row():
            user_query = gr.Textbox(label="User query", value="Мне нужен ноутбук до 70 тысяч для игр и учебы")
            context = gr.Textbox(label="Retrieved context JSON", value=default_context, lines=14)
        button = gr.Button("Generate structure")
        with gr.Row():
            raw_output = gr.Textbox(label="Raw model output", lines=12)
            parsed_output = gr.Code(label="Parsed JSON", language="json", lines=12)
            status = gr.Code(label="Validation status", language="json", lines=12)
        button.click(run, inputs=[user_query, context], outputs=[raw_output, parsed_output, status])

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
