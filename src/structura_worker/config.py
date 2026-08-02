from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


_WORKER_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class WorkerConfigurationError(RuntimeError):
    """A deterministic worker configuration error."""


def normalize_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise WorkerConfigurationError("Configured service origin is invalid")
    host = parsed.hostname.lower()
    port = parsed.port
    default_port = 443 if parsed.scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _allowed_origin(name: str, value: str) -> str:
    origin = normalize_origin(value)
    configured = os.environ.get(name, "")
    allowed = {
        normalize_origin(item)
        for item in configured.split(",")
        if item.strip()
    }
    if not allowed or origin not in allowed:
        raise WorkerConfigurationError("Configured service origin is not allowlisted")
    return origin


def _secret(path_value: str) -> str:
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise WorkerConfigurationError("Worker credential file is unavailable")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise WorkerConfigurationError("Worker credential file is unavailable") from exc
    if not value:
        raise WorkerConfigurationError("Worker credential file is empty")
    return value


def _bounded_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise WorkerConfigurationError(f"{name} is invalid") from exc
    if value < minimum or value > maximum:
        raise WorkerConfigurationError(f"{name} is outside the allowed range")
    return value


@dataclass(frozen=True)
class WorkerSettings:
    servermind_origin: str
    servermind_token: str
    mlflow_origin: str
    worker_id: str
    repository: str
    image_git_sha: str
    checkout_root: Path
    dataset_root: Path
    dataset_manifest: Path
    output_root: Path
    environment_lock: Path
    process_timeout_seconds: int
    heartbeat_seconds: int

    @classmethod
    def from_environment(cls) -> "WorkerSettings":
        worker_id = os.environ.get("SERVERMIND_WORKER_ID", "structura-worker-1")
        repository = os.environ.get(
            "STRUCTURA_REPOSITORY", "gotogrub/affective-alignment-lab"
        )
        image_git_sha = os.environ.get("STRUCTURA_IMAGE_GIT_SHA", "")
        if not _WORKER_ID_RE.fullmatch(worker_id):
            raise WorkerConfigurationError("Worker identity is invalid")
        if not _REPOSITORY_RE.fullmatch(repository):
            raise WorkerConfigurationError("Worker repository identity is invalid")
        if not re.fullmatch(r"^[0-9a-f]{40}$", image_git_sha):
            raise WorkerConfigurationError("Worker image commit identity is invalid")
        checkout_root = Path(os.environ.get("STRUCTURA_CHECKOUT_ROOT", "/workspace"))
        dataset_root = Path(os.environ.get("STRUCTURA_DATASET_ROOT", "/dataset"))
        dataset_manifest = Path(
            os.environ.get(
                "STRUCTURA_DATASET_MANIFEST",
                "/workspace/structura-dataset-manifest.sha256",
            )
        )
        output_root = Path(os.environ.get("STRUCTURA_OUTPUT_ROOT", "/artifacts/runs"))
        environment_lock = Path(
            os.environ.get(
                "STRUCTURA_ENVIRONMENT_LOCK", "/opt/structura-environment.lock"
            )
        )
        if not checkout_root.is_dir() or checkout_root.is_symlink():
            raise WorkerConfigurationError("Worker checkout is unavailable")
        if not dataset_root.is_dir() or dataset_root.is_symlink():
            raise WorkerConfigurationError("Worker dataset is unavailable")
        output_root.mkdir(parents=True, exist_ok=True)
        if output_root.is_symlink():
            raise WorkerConfigurationError("Worker output root is invalid")
        return cls(
            servermind_origin=_allowed_origin(
                "SERVERMIND_ALLOWED_ORIGINS", os.environ.get("SERVERMIND_URL", "")
            ),
            servermind_token=_secret(os.environ.get("SERVERMIND_WORKER_TOKEN_FILE", "")),
            mlflow_origin=_allowed_origin(
                "MLFLOW_ALLOWED_ORIGINS", os.environ.get("MLFLOW_TRACKING_URI", "")
            ),
            worker_id=worker_id,
            repository=repository,
            image_git_sha=image_git_sha,
            checkout_root=checkout_root.resolve(),
            dataset_root=dataset_root.resolve(),
            dataset_manifest=dataset_manifest.resolve(),
            output_root=output_root.resolve(),
            environment_lock=environment_lock.resolve(),
            process_timeout_seconds=_bounded_integer(
                "STRUCTURA_PROCESS_TIMEOUT_SECONDS", 86400, 60, 172800
            ),
            heartbeat_seconds=_bounded_integer(
                "STRUCTURA_HEARTBEAT_SECONDS", 300, 5, 1800
            ),
        )
