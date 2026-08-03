from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .config import WorkerConfigurationError, WorkerSettings


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class WorkerIntegrityError(RuntimeError):
    """Approved immutable inputs do not match the worker checkout."""


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise WorkerIntegrityError("Approved dependency lock is unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source_reference(
    settings: WorkerSettings, repository: str, commit_sha: str
) -> None:
    if repository != settings.repository or not _SHA_RE.fullmatch(commit_sha):
        raise WorkerIntegrityError("Approved source identity is invalid")


def verify_checkout(settings: WorkerSettings, repository: str, commit_sha: str) -> None:
    verify_source_reference(settings, repository, commit_sha)
    if settings.image_git_sha != commit_sha:
        raise WorkerIntegrityError("Worker image commit differs from approval")


def _dvc_dataset_hash(lock_path: Path) -> str:
    try:
        payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
        dependencies = payload["stages"]["audit-structura"]["deps"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise WorkerIntegrityError("DVC dataset identity is unavailable") from exc
    for dependency in dependencies:
        if dependency.get("path") == "data/structura/processed":
            value = dependency.get("md5")
            if isinstance(value, str):
                return value
    raise WorkerIntegrityError("DVC dataset identity is unavailable")


def verify_dataset(settings: WorkerSettings, dataset: dict[str, Any]) -> None:
    repository = dataset.get("repository")
    commit_sha = dataset.get("commit_sha")
    dvc_hash = dataset.get("dvc_hash")
    if not all(isinstance(value, str) for value in (repository, commit_sha, dvc_hash)):
        raise WorkerIntegrityError("Canonical dataset identity is incomplete")
    verify_source_reference(settings, repository, commit_sha)
    if _dvc_dataset_hash(settings.checkout_root / "dvc.lock") != dvc_hash:
        raise WorkerIntegrityError("Worker dataset differs from canonical DVC hash")
    _verify_dataset_manifest(settings)


def _verify_dataset_manifest(settings: WorkerSettings) -> None:
    if settings.dataset_manifest.is_symlink() or not settings.dataset_manifest.is_file():
        raise WorkerIntegrityError("Worker dataset manifest is unavailable")
    expected: dict[str, str] = {}
    for line in settings.dataset_manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise WorkerIntegrityError("Worker dataset manifest is invalid")
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.jsonl", name):
            raise WorkerIntegrityError("Worker dataset manifest path is invalid")
        expected[name] = digest
    dataset_entries = list(settings.dataset_root.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in dataset_entries):
        raise WorkerIntegrityError("Worker dataset contains an unsupported entry")
    actual_names = {item.name for item in dataset_entries}
    if not expected or actual_names != set(expected):
        raise WorkerIntegrityError("Worker dataset file set differs from manifest")
    for name, digest in expected.items():
        if file_hash(settings.dataset_root / name) != digest:
            raise WorkerIntegrityError("Worker dataset content differs from manifest")


def verify_training(
    settings: WorkerSettings,
    training_run: dict[str, Any],
    dataset: dict[str, Any],
    *,
    expected_status: str = "running",
) -> dict[str, Any]:
    payload = training_run.get("payload")
    if not isinstance(payload, dict):
        raise WorkerIntegrityError("Canonical training payload is unavailable")
    verify_dataset(settings, dataset)
    verify_checkout(settings, payload.get("repository", ""), payload.get("commit_sha", ""))
    if dataset.get("dvc_hash") != payload.get("dataset_dvc_hash"):
        raise WorkerIntegrityError("Approved training dataset differs from canonical dataset")
    config = payload.get("training_config")
    if not isinstance(config, dict) or canonical_hash(config) != payload.get("config_hash"):
        raise WorkerIntegrityError("Approved training configuration hash is invalid")
    if file_hash(settings.environment_lock) != payload.get("dependency_lock_hash"):
        raise WorkerIntegrityError("Worker dependency environment differs from approval")
    if payload.get("status") != expected_status:
        raise WorkerIntegrityError("Training run is not in the claimed state")
    return payload


def verify_completed_training(
    settings: WorkerSettings,
    training_run: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Verify immutable training lineage without requiring its historical image."""

    payload = training_run.get("payload")
    if not isinstance(payload, dict):
        raise WorkerIntegrityError("Canonical training payload is unavailable")
    verify_dataset(settings, dataset)
    verify_source_reference(
        settings, payload.get("repository", ""), payload.get("commit_sha", "")
    )
    if canonical_hash(payload) != training_run.get("payload_hash"):
        raise WorkerIntegrityError("Completed training payload hash is invalid")
    if dataset.get("dvc_hash") != payload.get("dataset_dvc_hash"):
        raise WorkerIntegrityError("Approved training dataset differs from canonical dataset")
    config = payload.get("training_config")
    if not isinstance(config, dict) or canonical_hash(config) != payload.get("config_hash"):
        raise WorkerIntegrityError("Approved training configuration hash is invalid")
    if payload.get("status") != "completed":
        raise WorkerIntegrityError("Training run is not completed")
    for field in (
        "repository",
        "commit_sha",
        "dataset_dvc_hash",
        "config_hash",
        "dependency_lock_hash",
    ):
        if training_run.get(field) != payload.get(field):
            raise WorkerIntegrityError("Completed training lineage is inconsistent")
    return payload


def verify_evaluator(settings: WorkerSettings, subject: dict[str, Any]) -> dict[str, Any]:
    evaluator = subject.get("evaluator")
    if not isinstance(evaluator, dict):
        raise WorkerIntegrityError("Canonical evaluator provenance is unavailable")
    if evaluator.get("protocol_version") != "1.1.0":
        raise WorkerIntegrityError("Evaluation protocol version is unsupported")
    verify_checkout(
        settings,
        evaluator.get("repository", ""),
        evaluator.get("commit_sha", ""),
    )
    dependency_hash = evaluator.get("dependency_lock_hash")
    if not isinstance(dependency_hash, str) or file_hash(
        settings.environment_lock
    ) != dependency_hash:
        raise WorkerIntegrityError("Evaluator dependency environment differs from request")
    image_digest = evaluator.get("image_digest")
    if (
        not isinstance(image_digest, str)
        or not _DIGEST_RE.fullmatch(image_digest)
        or image_digest != settings.image_digest
    ):
        raise WorkerIntegrityError("Evaluator image digest differs from request")
    return evaluator


def safe_run_directory(settings: WorkerSettings, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", run_id):
        raise WorkerConfigurationError("Run identity is invalid")
    path = (settings.output_root / run_id).resolve()
    if settings.output_root not in path.parents:
        raise WorkerConfigurationError("Run output escaped the configured root")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise WorkerConfigurationError("Run output directory is invalid")
    return path
