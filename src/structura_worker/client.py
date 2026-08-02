from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class WorkerProtocolError(RuntimeError):
    """A bounded error returned by the ServerMind worker boundary."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), port


class ServerMindClient:
    def __init__(self, origin: str, token: str, worker_id: str) -> None:
        self._origin = origin.rstrip("/")
        self._expected_origin = _origin(origin)
        self._token = token
        self._worker_id = worker_id
        self._opener = build_opener(_NoRedirect())

    def _read(self, request: Request) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            with self._opener.open(request, timeout=15) as response:
                if _origin(response.geturl()) != self._expected_origin:
                    raise WorkerProtocolError("ServerMind response origin is invalid")
                if 300 <= response.status < 400:
                    raise WorkerProtocolError("ServerMind redirect was rejected")
                raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise WorkerProtocolError("ServerMind response exceeded the size limit")
        except HTTPError as exc:
            if 300 <= exc.code < 400:
                raise WorkerProtocolError("ServerMind redirect was rejected") from None
            raise WorkerProtocolError(
                f"ServerMind request failed with status {exc.code}"
            ) from None
        except URLError:
            raise WorkerProtocolError("ServerMind request is unavailable") from None
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise WorkerProtocolError("ServerMind response was invalid") from None
        if not isinstance(result, (dict, list)):
            raise WorkerProtocolError("ServerMind response was invalid")
        return result

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        url = f"{self._origin}/{path.lstrip('/')}"
        if _origin(url) != self._expected_origin:
            raise WorkerProtocolError("ServerMind request origin is invalid")
        request = Request(
            url,
            data=json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
                "X-ServerMind-Worker-Token": self._token,
            },
        )
        result = self._read(request)
        if not isinstance(result, dict):
            raise WorkerProtocolError("ServerMind response was invalid")
        return result

    def queued_work(self) -> list[dict[str, Any]]:
        url = f"{self._origin}/api/lab/work/queued"
        if _origin(url) != self._expected_origin:
            raise WorkerProtocolError("ServerMind request origin is invalid")
        result = self._read(
            Request(
                url,
                method="GET",
                headers={"X-ServerMind-Worker-Token": self._token},
            )
        )
        if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
            raise WorkerProtocolError("ServerMind response was invalid")
        return result

    def claim(self, kind: str, run_id: str, operation_id: str) -> dict[str, Any]:
        return self._post(
            f"api/lab/{kind}-runs/{run_id}/claim",
            {"worker_id": self._worker_id},
            f"worker-claim:{kind}:{run_id}:{operation_id}",
        )

    def heartbeat(
        self,
        kind: str,
        run_id: str,
        fence_token: int,
        sequence: int,
        operation_id: str,
    ) -> dict[str, Any]:
        return self._post(
            f"api/lab/{kind}-runs/{run_id}/heartbeat",
            {"worker_id": self._worker_id, "fence_token": fence_token},
            f"worker-heartbeat:{kind}:{run_id}:{operation_id}:{sequence}",
        )

    def complete(
        self,
        kind: str,
        run_id: str,
        fence_token: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        body = {"fence_token": fence_token, **payload}
        completion_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(body, sort_keys=True, separators=(",", ":")),
        ).hex
        return self._post(
            f"api/lab/{kind}-runs/{run_id}/complete",
            body,
            f"worker-complete:{kind}:{run_id}:{fence_token}:{completion_id}",
        )
