from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Iterator


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger("structura")


def log_json(logger: logging.Logger, message: str, payload: Any, *, level: int = logging.INFO) -> None:
    logger.log(level, "%s: %s", message, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


@contextmanager
def log_step(logger: logging.Logger, name: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("START %s", name)
    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.2fs", name, time.perf_counter() - start)
        raise
    logger.info("DONE  %s in %.2fs", name, time.perf_counter() - start)


def summarize_lengths(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "max": 0}
    p95 = quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
    return {
        "min": min(values),
        "median": median(values),
        "mean": round(mean(values), 2),
        "p95": p95,
        "max": max(values),
    }


def disk_usage_summary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    target = path if path.exists() else path.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "total_gb": round(usage.total / 1024**3, 2),
        "used_gb": round(usage.used / 1024**3, 2),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def runtime_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
