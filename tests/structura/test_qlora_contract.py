from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_qlora_config_pins_immutable_model_revision() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/structura/train_qwen3_4b_qlora.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["base_model"]["id"] == "Qwen/Qwen3-4B"
    assert re.fullmatch(r"[0-9a-f]{40}", config["base_model"]["revision"])
    assert config["training"]["seed"] == 42
    assert config["qlora"]["quant_type"] == "nf4"
    assert config["qlora"]["rank"] > 0


def test_worker_image_records_full_dependency_environment() -> None:
    dockerfile = (ROOT / "docker/structura-worker.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "pip freeze --all > /opt/structura-environment.lock" in dockerfile
    assert "COPY data/" not in dockerfile
