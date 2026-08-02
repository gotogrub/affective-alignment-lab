from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from structura_worker.client import ServerMindClient, WorkerProtocolError
from structura_worker.config import WorkerSettings
from structura_worker.execution import failure_payload
from structura_worker.integrity import WorkerIntegrityError, verify_dataset


def _server(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_worker_client_rejects_redirect_without_forwarding_token(
    status_code: int,
) -> None:
    credential = "worker_secret_that_must_not_escape"
    target_requests: list[dict[str, str]] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            target_requests.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, format: str, *args: object) -> None:
            return

    target, target_thread = _server(TargetHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(status_code)
            self.send_header(
                "Location", f"http://127.0.0.1:{target.server_port}/collect"
            )
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    source, source_thread = _server(RedirectHandler)
    try:
        client = ServerMindClient(
            f"http://127.0.0.1:{source.server_port}", credential, "worker-1"
        )
        with pytest.raises(WorkerProtocolError) as captured:
            client.claim("training", "01234567-89ab-cdef-0123-456789abcdef", "op")
        assert target_requests == []
        assert credential not in str(captured.value)
    finally:
        source.shutdown()
        target.shutdown()
        source_thread.join(timeout=2)
        target_thread.join(timeout=2)


def test_worker_verifies_embedded_commit_and_dataset_without_git(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "output"
    checkout.mkdir()
    dataset_root.mkdir()
    output.mkdir()
    (checkout / "dvc.lock").write_text(
        """schema: '2.0'
stages:
  audit-structura:
    deps:
      - path: data/structura/processed
        md5: dataset-hash.dir
""",
        encoding="utf-8",
    )
    content = b'{"id":"one"}\n'
    (dataset_root / "test.jsonl").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = checkout / "structura-dataset-manifest.sha256"
    manifest.write_text(f"{digest}  test.jsonl\n", encoding="utf-8")
    environment_lock = checkout / "environment.lock"
    environment_lock.write_text("package==1\n", encoding="utf-8")
    commit = "a" * 40
    settings = WorkerSettings(
        servermind_origin="http://servermind",
        servermind_token="secret",
        mlflow_origin="http://mlflow",
        worker_id="worker-1",
        repository="gotogrub/affective-alignment-lab",
        image_git_sha=commit,
        checkout_root=checkout,
        dataset_root=dataset_root,
        dataset_manifest=manifest,
        output_root=output,
        environment_lock=environment_lock,
        process_timeout_seconds=60,
        heartbeat_seconds=5,
    )
    verify_dataset(
        settings,
        {
            "repository": "gotogrub/affective-alignment-lab",
            "commit_sha": commit,
            "dvc_hash": "dataset-hash.dir",
        },
    )
    (dataset_root / "test.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(WorkerIntegrityError, match="content"):
        verify_dataset(
            settings,
            {
                "repository": "gotogrub/affective-alignment-lab",
                "commit_sha": commit,
                "dvc_hash": "dataset-hash.dir",
            },
        )


def test_worker_rejects_symlink_in_dataset(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    dataset_root = tmp_path / "dataset"
    output = tmp_path / "output"
    checkout.mkdir()
    dataset_root.mkdir()
    output.mkdir()
    (checkout / "dvc.lock").write_text(
        """schema: '2.0'
stages:
  audit-structura:
    deps:
      - path: data/structura/processed
        md5: dataset-hash.dir
""",
        encoding="utf-8",
    )
    content = b'{"id":"one"}\n'
    source = tmp_path / "source.jsonl"
    source.write_bytes(content)
    (dataset_root / "test.jsonl").symlink_to(source)
    digest = hashlib.sha256(content).hexdigest()
    manifest = checkout / "structura-dataset-manifest.sha256"
    manifest.write_text(f"{digest}  test.jsonl\n", encoding="utf-8")
    environment_lock = checkout / "environment.lock"
    environment_lock.write_text("package==1\n", encoding="utf-8")
    commit = "a" * 40
    settings = WorkerSettings(
        servermind_origin="http://servermind",
        servermind_token="secret",
        mlflow_origin="http://mlflow",
        worker_id="worker-1",
        repository="gotogrub/affective-alignment-lab",
        image_git_sha=commit,
        checkout_root=checkout,
        dataset_root=dataset_root,
        dataset_manifest=manifest,
        output_root=output,
        environment_lock=environment_lock,
        process_timeout_seconds=60,
        heartbeat_seconds=5,
    )

    with pytest.raises(WorkerIntegrityError, match="unsupported entry"):
        verify_dataset(
            settings,
            {
                "repository": "gotogrub/affective-alignment-lab",
                "commit_sha": commit,
                "dvc_hash": "dataset-hash.dir",
            },
        )


def test_worker_failure_payload_never_persists_exception_text() -> None:
    credential = "hf_secret_that_must_not_persist"
    payload = failure_payload(RuntimeError(f"Bearer {credential}"))
    assert credential not in json.dumps(payload)
    assert payload["error_code"] == "WORKER_EXECUTION_FAILED"


def test_dataset_manifest_matches_tracked_structura_data() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = root / "structura-dataset-manifest.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        data = root / "data/structura/processed" / name
        assert hashlib.sha256(data.read_bytes()).hexdigest() == digest
