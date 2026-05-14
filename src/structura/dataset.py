from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import StructuraSample, dump_model, validate_model


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=indent)
        file.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")


def load_samples(path: str | Path) -> list[StructuraSample]:
    return [validate_model(StructuraSample, record) for record in read_jsonl(path)]


def samples_to_records(samples: Iterable[StructuraSample]) -> list[dict[str, Any]]:
    return [dump_model(sample) for sample in samples]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def split_group_key(record: dict[str, Any], *, group_by: str = "input_target") -> str:
    if group_by == "input":
        payload = record["input"]
    elif group_by == "target":
        payload = record["target"]
    elif group_by == "input_target":
        payload = {"input": record["input"], "target": record["target"]}
    else:
        raise ValueError(f"Unknown group_by value: {group_by}")
    return canonical_json(payload)


def _split_group_counts(total_groups: int, train_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    if total_groups <= 0:
        return 0, 0, 0
    if total_groups == 1:
        return 1, 0, 0
    if total_groups == 2:
        return 1, 0, 1

    test_ratio = 1.0 - train_ratio - valid_ratio
    valid_count = max(1, round(total_groups * valid_ratio)) if valid_ratio > 0 else 0
    test_count = max(1, round(total_groups * test_ratio))

    while valid_count + test_count >= total_groups:
        if valid_count >= test_count and valid_count > 0:
            valid_count -= 1
        else:
            test_count -= 1

    train_count = total_groups - valid_count - test_count
    return train_count, valid_count, test_count


def split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    group_by: str = "input_target",
    stratify_key: str = "scenario",
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be below 1")

    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        stratum = str(record.get(stratify_key) or "unknown")
        grouped[stratum][split_group_key(record, group_by=group_by)].append(record)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for stratum in sorted(grouped):
        groups = list(grouped[stratum].values())
        rng.shuffle(groups)
        train_count, valid_count, _ = _split_group_counts(len(groups), train_ratio, valid_ratio)

        train_groups = groups[:train_count]
        valid_groups = groups[train_count : train_count + valid_count]
        test_groups = groups[train_count + valid_count :]

        for group in train_groups:
            splits["train"].extend(group)
        for group in valid_groups:
            splits["valid"].extend(group)
        for group in test_groups:
            splits["test"].extend(group)

    for split_name, split_records_ in splits.items():
        rng.shuffle(split_records_)
        for record in split_records_:
            record["split"] = split_name

    return splits
