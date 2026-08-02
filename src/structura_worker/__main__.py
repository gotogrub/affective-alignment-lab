from __future__ import annotations

import argparse
import json
import time

from .client import ServerMindClient
from .config import WorkerSettings
from .execution import StructuraWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one canonical Structura job")
    parser.add_argument("kind", choices=["evaluation", "training", "watch"])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    settings = WorkerSettings.from_environment()
    client = ServerMindClient(
        settings.servermind_origin,
        settings.servermind_token,
        settings.worker_id,
    )
    worker = StructuraWorker(settings, client)
    if args.kind == "watch":
        _watch(worker, client, once=args.once, poll_seconds=args.poll_seconds)
        return
    if not args.run_id:
        parser.error("run_id is required for a single job")
    result = worker.run(args.kind, args.run_id)
    print(json.dumps({"state": _state(result)}, sort_keys=True))


def _state(result: dict) -> str:
    for key in ("training_run", "evaluation_run"):
        value = result.get(key)
        if isinstance(value, dict) and isinstance(value.get("state"), str):
            return value["state"]
    return "unchanged"


def _watch(
    worker: StructuraWorker,
    client: ServerMindClient,
    *,
    once: bool,
    poll_seconds: int,
) -> None:
    if poll_seconds < 5 or poll_seconds > 300:
        raise RuntimeError("Poll interval is outside the allowed range")
    while True:
        work = client.queued_work()
        if work:
            item = work[0]
            kind = item.get("kind")
            run_id = item.get("run_id")
            if kind not in {"evaluation", "training"} or not isinstance(run_id, str):
                raise RuntimeError("Queued work response is invalid")
            result = worker.run(kind, run_id)
            print(json.dumps({"kind": kind, "run_id": run_id, "state": _state(result)}, sort_keys=True))
        if once:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
