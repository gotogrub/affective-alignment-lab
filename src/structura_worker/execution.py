from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .client import ServerMindClient, WorkerProtocolError
from .config import WorkerConfigurationError, WorkerSettings
from .integrity import (
    WorkerIntegrityError,
    safe_run_directory,
    verify_completed_training,
    verify_dataset,
    verify_evaluator,
    verify_training,
)


_MAX_LOG_BYTES = 10 * 1024 * 1024
_SCALAR_METRICS = {
    "intent_accuracy",
    "answer_type_accuracy",
    "product_selection_precision",
    "product_selection_recall",
    "product_selection_f1",
    "hallucinated_product_id_rate",
    "clarification_accuracy",
    "handoff_accuracy",
    "semantic_correctness",
    "valid_json_rate",
    "schema_valid_rate",
    "latency_ms_p50",
    "latency_ms_p95",
}


class WorkerExecutionError(RuntimeError):
    """A bounded worker execution failure."""


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    heartbeat_seconds: int,
    heartbeat: Callable[[], None],
    environment: dict[str, str] | None = None,
) -> None:
    started = time.monotonic()
    next_heartbeat = started + heartbeat_seconds
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now - started > timeout_seconds:
                    raise TimeoutError("Worker subprocess timed out")
                if stdout.tell() > _MAX_LOG_BYTES or stderr.tell() > _MAX_LOG_BYTES:
                    raise WorkerExecutionError("Worker subprocess log limit exceeded")
                if now >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = now + heartbeat_seconds
                time.sleep(1)
        except Exception:
            _terminate(process)
            raise
    if process.returncode != 0:
        raise WorkerExecutionError("Worker subprocess failed")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 5_000_000:
        raise WorkerExecutionError("Worker artifact is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerExecutionError("Worker artifact is invalid") from exc
    if not isinstance(payload, dict):
        raise WorkerExecutionError("Worker artifact is invalid")
    return payload


def _scalar_metrics(payload: dict[str, Any]) -> dict[str, float | int | bool | str | None]:
    return {
        key: value
        for key, value in payload.items()
        if key in _SCALAR_METRICS and isinstance(value, (str, int, float, bool, type(None)))
    }


def _evaluation_status(
    metrics: dict[str, float | int | bool | str | None],
    thresholds: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    failed_thresholds = [
        name
        for name, threshold in thresholds.items()
        if not isinstance(metrics.get(name), (int, float))
        or float(metrics[name]) < threshold
    ]
    safety_findings: list[str] = []
    if float(metrics.get("schema_valid_rate") or 0) < 1:
        safety_findings.append("schema_validity_regression")
    if float(metrics.get("hallucinated_product_id_rate") or 0) > 0:
        safety_findings.append("hallucinated_product_identifier")
    status = "passed" if not failed_thresholds and not safety_findings else "failed"
    return status, {
        "status": status,
        "findings": [*failed_thresholds, *safety_findings],
    }


class StructuraWorker:
    def __init__(self, settings: WorkerSettings, client: ServerMindClient) -> None:
        self.settings = settings
        self.client = client

    def run(self, kind: str, run_id: str) -> dict[str, Any]:
        if kind not in {"evaluation", "training"}:
            raise WorkerConfigurationError("Worker job kind is invalid")
        operation_id = uuid.uuid4().hex
        claim = self.client.claim(kind, run_id, operation_id)
        if claim.get("execute") is not True:
            return claim
        fence_token = claim.get("fence_token")
        if not isinstance(fence_token, int):
            raise WorkerProtocolError("ServerMind claim omitted the fencing token")
        heartbeat_sequence = 0

        def heartbeat() -> None:
            nonlocal heartbeat_sequence
            heartbeat_sequence += 1
            self.client.heartbeat(
                kind,
                run_id,
                fence_token,
                heartbeat_sequence,
                operation_id,
            )

        try:
            aggregate = claim.get(f"{kind}_run")
            if isinstance(aggregate, dict) and aggregate.get("previous_gpu_profile") == "serve":
                raise WorkerConfigurationError(
                    "Serving runtime must be unloaded by the host operator"
                )
            if kind == "training":
                completion = self._training(run_id, claim, heartbeat)
            else:
                completion = self._evaluation(run_id, claim, heartbeat)
        except Exception as exc:
            completion = failure_payload(exc)
        return self.client.complete(kind, run_id, fence_token, completion)

    def _training(
        self,
        run_id: str,
        claim: dict[str, Any],
        heartbeat: Callable[[], None],
    ) -> dict[str, Any]:
        training_run = claim.get("training_run")
        execution_input = claim.get("execution_input")
        if not isinstance(training_run, dict) or not isinstance(execution_input, dict):
            raise WorkerProtocolError("Training claim omitted immutable execution input")
        dataset = execution_input.get("dataset")
        if not isinstance(dataset, dict):
            raise WorkerProtocolError("Training claim omitted the canonical dataset")
        approved = verify_training(self.settings, training_run, dataset)
        run_dir = safe_run_directory(self.settings, run_id)
        payload_path = run_dir / "approved-training-run.json"
        payload_path.write_text(
            json.dumps(approved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stdout_path = run_dir / "train.stdout.log"
        stderr_path = run_dir / "train.stderr.log"
        environment = {
            **os.environ,
            "TRAINING_GIT_SHA": approved["commit_sha"],
            "DATASET_DVC_HASH": approved["dataset_dvc_hash"],
            "DEPENDENCY_LOCK_HASH": approved["dependency_lock_hash"],
            "MLFLOW_TRACKING_URI": self.settings.mlflow_origin,
            "MLFLOW_RUN_NAME": f"structura-{training_run['training_run_key']}",
        }
        heartbeat()
        run_process(
            [
                sys.executable,
                "scripts/structura/train_qlora.py",
                "--approved-run",
                str(payload_path),
                "--output-dir",
                str(run_dir / "training"),
            ],
            cwd=self.settings.checkout_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=self.settings.process_timeout_seconds,
            heartbeat_seconds=self.settings.heartbeat_seconds,
            heartbeat=heartbeat,
            environment=environment,
        )
        heartbeat()
        manifest = _read_json(run_dir / "training" / "training-manifest.json")
        if (
            manifest.get("repository_commit") != approved["commit_sha"]
            or manifest.get("dataset_dvc_hash") != approved["dataset_dvc_hash"]
            or manifest.get("config_hash") != approved["config_hash"]
            or manifest.get("dependency_lock_hash") != approved["dependency_lock_hash"]
        ):
            raise WorkerIntegrityError("Training output provenance differs from approval")
        mlflow_run_id = manifest.get("mlflow_run_id")
        adapter_uri = manifest.get("adapter_uri")
        if not isinstance(mlflow_run_id, str) or not isinstance(adapter_uri, str):
            raise WorkerExecutionError("Training output references are incomplete")
        self._log_worker_files(mlflow_run_id, [stdout_path, stderr_path])
        return {
            "outcome": "completed",
            "metrics": _scalar_metrics(manifest.get("metrics", {})),
            "output_artifacts": {
                "adapter_uri": adapter_uri,
                "checkpoint_uri": None,
                "mlflow_run_id": mlflow_run_id,
                "logs_uri": f"mlflow://{mlflow_run_id}/artifacts/worker-logs",
            },
        }

    def _evaluation(
        self,
        run_id: str,
        claim: dict[str, Any],
        heartbeat: Callable[[], None],
    ) -> dict[str, Any]:
        evaluation_run = claim.get("evaluation_run")
        execution_input = claim.get("execution_input")
        if not isinstance(evaluation_run, dict) or not isinstance(execution_input, dict):
            raise WorkerProtocolError("Evaluation claim omitted immutable execution input")
        dataset = execution_input.get("dataset")
        experiment = execution_input.get("experiment")
        if not isinstance(dataset, dict) or not isinstance(experiment, dict):
            raise WorkerProtocolError("Evaluation claim omitted canonical provenance")
        run_dir = safe_run_directory(self.settings, run_id)
        subject = evaluation_run.get("subject")
        if not isinstance(subject, dict):
            raise WorkerProtocolError("Evaluation subject is unavailable")
        training_run = execution_input.get("training_run")
        adapter: str | None = None
        if evaluation_run.get("kind") == "training":
            if not isinstance(training_run, dict):
                raise WorkerProtocolError("Candidate evaluation omitted training lineage")
            verify_evaluator(self.settings, subject)
            approved = verify_completed_training(self.settings, training_run, dataset)
            if (
                subject.get("revision") != training_run.get("payload_hash")
                or subject.get("model_id") != training_run.get("base_model_id")
            ):
                raise WorkerIntegrityError(
                    "Evaluation subject differs from completed training lineage"
                )
            artifacts = approved.get("output_artifacts")
            if not isinstance(artifacts, dict) or not isinstance(
                artifacts.get("adapter_uri"), str
            ):
                raise WorkerIntegrityError("Approved adapter reference is unavailable")
            adapter = self._download_artifact(artifacts["adapter_uri"], run_dir / "adapter")
        else:
            verify_dataset(self.settings, dataset)
        return self._run_evaluation_process(
            run_id,
            run_dir,
            evaluation_run,
            experiment,
            dataset,
            subject,
            adapter,
            heartbeat,
        )

    def _run_evaluation_process(
        self,
        run_id: str,
        run_dir: Path,
        evaluation_run: dict[str, Any],
        experiment: dict[str, Any],
        dataset: dict[str, Any],
        subject: dict[str, Any],
        adapter: str | None,
        heartbeat: Callable[[], None],
    ) -> dict[str, Any]:
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_origin)
        output_dir = run_dir / "evaluation"
        stdout_path = run_dir / "evaluation.stdout.log"
        stderr_path = run_dir / "evaluation.stderr.log"
        rules_dir = run_dir / "rules-baseline"
        rules_stdout = run_dir / "rules.stdout.log"
        rules_stderr = run_dir / "rules.stderr.log"
        command = [
            sys.executable,
            "scripts/structura/evaluate_causal.py",
            "--model",
            subject["model_id"],
            "--revision",
            (
                experiment["spec_payload"]["baseline"]["base_model_revision"]
                if evaluation_run.get("kind") == "training"
                else subject["revision"]
            ),
            "--test-path",
            str(self.settings.dataset_root / "test.jsonl"),
            "--output-dir",
            str(output_dir),
        ]
        if adapter:
            command.extend(["--adapter", adapter])
        with mlflow.start_run(run_name=f"structura-evaluation-{run_id}") as active:
            mlflow.log_params(
                {
                    "repository_commit": dataset["commit_sha"],
                    "dataset_dvc_hash": dataset["dvc_hash"],
                    "subject_model_id": subject["model_id"],
                    "subject_revision": subject["revision"],
                    "evaluation_kind": evaluation_run["kind"],
                    "evaluator_repository": (
                        subject.get("evaluator", {}).get("repository")
                    ),
                    "evaluator_commit": subject.get("evaluator", {}).get("commit_sha"),
                    "evaluator_image_digest": (
                        subject.get("evaluator", {}).get("image_digest")
                    ),
                }
            )
            rules_metrics: dict[str, float | int | bool | str | None] = {}
            if evaluation_run.get("kind") == "baseline":
                run_process(
                    [
                        sys.executable,
                        "scripts/structura/evaluate.py",
                        "--baseline",
                        "rules",
                        "--test-path",
                        str(self.settings.dataset_root / "test.jsonl"),
                        "--output-predictions",
                        str(rules_dir / "predictions.jsonl"),
                        "--output-metrics",
                        str(rules_dir / "metrics.json"),
                    ],
                    cwd=self.settings.checkout_root,
                    stdout_path=rules_stdout,
                    stderr_path=rules_stderr,
                    timeout_seconds=self.settings.process_timeout_seconds,
                    heartbeat_seconds=self.settings.heartbeat_seconds,
                    heartbeat=heartbeat,
                    environment=os.environ.copy(),
                )
                rules_metrics = {
                    f"rules_{key}": value
                    for key, value in _scalar_metrics(
                        _read_json(rules_dir / "metrics.json")
                    ).items()
                }
            heartbeat()
            run_process(
                command,
                cwd=self.settings.checkout_root,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=self.settings.process_timeout_seconds,
                heartbeat_seconds=self.settings.heartbeat_seconds,
                heartbeat=heartbeat,
                environment={**os.environ, "MLFLOW_TRACKING_URI": self.settings.mlflow_origin},
            )
            heartbeat()
            metrics = {
                **_scalar_metrics(_read_json(output_dir / "metrics.json")),
                **rules_metrics,
            }
            mlflow.log_metrics(
                {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
            )
            mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
            if rules_metrics:
                mlflow.log_artifacts(str(rules_dir), artifact_path="rules-baseline")
                mlflow.log_artifact(str(rules_stdout), artifact_path="worker-logs")
                mlflow.log_artifact(str(rules_stderr), artifact_path="worker-logs")
            mlflow.log_artifact(str(stdout_path), artifact_path="worker-logs")
            mlflow.log_artifact(str(stderr_path), artifact_path="worker-logs")
            mlflow_run_id = active.info.run_id
        thresholds = experiment["spec_payload"]["evaluation_plan"][
            "promotion_thresholds"
        ]
        status, safety = _evaluation_status(metrics, thresholds)
        return {
            "outcome": "completed",
            "status": status,
            "metrics": metrics,
            "thresholds": thresholds,
            "safety": safety,
            "artifact_refs": {
                "report_uri": f"mlflow://{mlflow_run_id}/artifacts/evaluation/metrics.json",
                "predictions_uri": f"mlflow://{mlflow_run_id}/artifacts/evaluation/predictions.jsonl",
                "mlflow_run_id": mlflow_run_id,
            },
        }

    def _download_artifact(self, uri: str, destination: Path) -> str:
        import mlflow

        if not uri.startswith(("mlflow-artifacts:/", "runs:/")):
            raise WorkerIntegrityError("Adapter artifact origin is not allowlisted")
        destination.mkdir(parents=True, exist_ok=True)
        resolved = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=uri, dst_path=str(destination)
            )
        ).resolve()
        if destination.resolve() not in resolved.parents and resolved != destination.resolve():
            raise WorkerIntegrityError("Adapter artifact escaped the worker output root")
        return str(resolved)

    def _log_worker_files(self, run_id: str, paths: list[Path]) -> None:
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_origin)
        with mlflow.start_run(run_id=run_id):
            for path in paths:
                mlflow.log_artifact(str(path), artifact_path="worker-logs")


def failure_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, WorkerIntegrityError):
        code, error_class, message, retryable = (
            "INPUT_INTEGRITY_FAILED",
            "WorkerIntegrityError",
            "Approved immutable inputs failed verification",
            False,
        )
    elif isinstance(exc, WorkerConfigurationError):
        code, error_class, message, retryable = (
            "WORKER_CONFIGURATION_FAILED",
            "WorkerConfigurationError",
            "Laboratory worker configuration is invalid",
            False,
        )
    elif isinstance(exc, WorkerProtocolError):
        code, error_class, message, retryable = (
            "CONTROL_PLANE_UNAVAILABLE",
            "WorkerProtocolError",
            "Laboratory control plane request failed",
            True,
        )
    elif isinstance(exc, TimeoutError):
        code, error_class, message, retryable = (
            "WORKER_TIMEOUT",
            "TimeoutError",
            "Laboratory worker exceeded the execution timeout",
            True,
        )
    else:
        code, error_class, message, retryable = (
            "WORKER_EXECUTION_FAILED",
            type(exc).__name__[:128],
            "Laboratory worker execution failed",
            True,
        )
    return {
        "outcome": "failed",
        "metrics": {},
        "error_code": code,
        "error_class": error_class,
        "safe_message": message,
        "retryable": retryable,
    }
