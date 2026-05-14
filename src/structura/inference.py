from __future__ import annotations

from pathlib import Path
from typing import Any

from .baselines import rules_baseline
from .formatting import format_prompt
from .validators import validate_output


class StructuraGenerator:
    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        template: str = "instruction",
        device: int | str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.checkpoint = checkpoint
        self.template = template
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.device = device

        if checkpoint:
            self._load(checkpoint, device=device)

    def _load(self, checkpoint: str, *, device: int | str | None) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Install torch and transformers to run model inference") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

    def generate(self, user_query: str, retrieved_context: list[dict[str, Any]]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            prediction = rules_baseline(user_query, retrieved_context)
            return {"raw_output": prediction, "parsed": prediction, "validation": validate_output(prediction)}

        prompt = format_prompt({"user_query": user_query, "retrieved_context": retrieved_context}, template=self.template)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False)
        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        validation = validate_output(raw_output)
        return {"raw_output": raw_output, "parsed": validation.parsed, "validation": validation}


def load_generator(checkpoint: str | Path | None = None, **kwargs: Any) -> StructuraGenerator:
    checkpoint_str = str(checkpoint) if checkpoint else None
    return StructuraGenerator(checkpoint_str, **kwargs)
