# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared, side-effect-free RH20T cfg4 parsing and resampling helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SOURCE_ROOT = Path("/mnt/pfs/swy/dataset/RH20T/RH20T_cfg4")
DEFAULT_DERIVED_ROOT = Path("/mnt/pfs/swy/dataset/RH20T/derived/stage2_ur5_joint_v1")
DEFAULT_TASK_DESCRIPTIONS = Path("/mnt/pfs/swy/dataset/RH20T/task_description.json")

EXTERIOR_CAMERAS = ("104122062295", "104122062823")
PRIMARY_SERIAL = EXTERIOR_CAMERAS[0]
FPS = 15
CHUNK_LENGTH = 32
GRIPPER_MAX_WIDTH_MM = 85.0
ACTION_LAYOUT = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper_close_fraction",
)

_EPISODE_RE = re.compile(r"^(task_\d{4})_(user_\d{4})_(scene_\d{4})_cfg_(\d{4})(?P<human>_human(?:_\d+)?)?$")


@dataclass(frozen=True, slots=True)
class EpisodeIdentity:
    task_id: str
    user_id: str
    scene_id: str
    cfg: int
    is_human: bool

    @property
    def group_key(self) -> str:
        return f"{self.task_id}|{self.scene_id}|{self.user_id}"

    @property
    def scene_number(self) -> int:
        return int(self.scene_id.removeprefix("scene_"))


def parse_episode_name(name: str) -> EpisodeIdentity:
    match = _EPISODE_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported RH20T episode directory name: {name!r}")
    return EpisodeIdentity(
        task_id=match.group(1),
        user_id=match.group(2),
        scene_id=match.group(3),
        cfg=int(match.group(4)),
        is_human=match.group("human") is not None,
    )


def load_object_npy(path: Path) -> Any:
    value = np.load(path, allow_pickle=True)
    return value.item() if value.shape == () else value


def require_serial_mapping(path: Path, serial: str) -> dict[Any, Any]:
    value = load_object_npy(path)
    if not isinstance(value, dict) or not isinstance(value.get(serial), dict):
        raise ValueError(f"{path} has no mapping for serial {serial}")
    return value[serial]


def load_joint_stream(path: Path, serial: str, finish_time_ms: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows = require_serial_mapping(path, serial)
    ordered = sorted((float(timestamp), np.asarray(joint, dtype=np.float64)) for timestamp, joint in rows.items())
    if finish_time_ms is not None:
        ordered = [item for item in ordered if item[0] <= finish_time_ms]
    if not ordered:
        raise ValueError(f"No joint samples remain in {path} for serial {serial}")
    timestamps = np.asarray([item[0] for item in ordered], dtype=np.float64)
    values = np.stack([item[1] for item in ordered])
    if values.shape[1:] != (6,) or not np.isfinite(values).all():
        raise ValueError(f"Joint stream must be finite [N,6], got {values.shape} in {path}")
    require_strictly_increasing(timestamps, f"joint:{path}")
    return timestamps, values


def load_gripper_width_stream(
    path: Path, serial: str, finish_time_ms: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    rows = require_serial_mapping(path, serial)
    ordered: list[tuple[float, float]] = []
    for timestamp, row in rows.items():
        if not isinstance(row, dict) or "gripper_command" not in row:
            raise ValueError(f"Malformed gripper row at {timestamp} in {path}")
        command = np.asarray(row["gripper_command"], dtype=np.float64).reshape(-1)
        if command.size < 1 or not np.isfinite(command[0]):
            raise ValueError(f"Malformed gripper_command at {timestamp} in {path}")
        ordered.append((float(timestamp), float(command[0])))
    ordered.sort()
    if finish_time_ms is not None:
        ordered = [item for item in ordered if item[0] <= finish_time_ms]
    if not ordered:
        raise ValueError(f"No gripper samples remain in {path} for serial {serial}")
    timestamps = np.asarray([item[0] for item in ordered], dtype=np.float64)
    width_mm = np.asarray([item[1] for item in ordered], dtype=np.float64)
    require_strictly_increasing(timestamps, f"gripper:{path}")
    if not np.isfinite(width_mm).all():
        raise ValueError(f"Non-finite gripper width in {path}")
    return timestamps, width_mm


def load_tcp_base_stream(path: Path, serial: str, finish_time_ms: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    value = load_object_npy(path)
    if not isinstance(value, dict) or not isinstance(value.get(serial), list):
        raise ValueError(f"{path} has no TCP list for serial {serial}")
    ordered = sorted((float(row["timestamp"]), np.asarray(row["tcp"], dtype=np.float64)) for row in value[serial])
    if finish_time_ms is not None:
        ordered = [item for item in ordered if item[0] <= finish_time_ms]
    if not ordered:
        raise ValueError(f"No TCP samples remain in {path} for serial {serial}")
    timestamps = np.asarray([item[0] for item in ordered], dtype=np.float64)
    values = np.stack([item[1] for item in ordered])
    if values.shape[1:] != (7,) or not np.isfinite(values).all():
        raise ValueError(f"TCP stream must be finite [N,7], got {values.shape} in {path}")
    require_strictly_increasing(timestamps, f"tcp:{path}")
    return timestamps, values


def load_color_timestamps(path: Path, finish_time_ms: float | None = None) -> np.ndarray:
    value = load_object_npy(path)
    if isinstance(value, dict):
        value = value.get("color")
    timestamps = np.asarray(value, dtype=np.float64).reshape(-1)
    if finish_time_ms is not None:
        timestamps = timestamps[timestamps <= finish_time_ms]
    if timestamps.size == 0:
        raise ValueError(f"No color timestamps remain in {path}")
    require_strictly_increasing(timestamps, f"camera:{path}")
    return timestamps


def require_strictly_increasing(timestamps: np.ndarray, label: str) -> None:
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.isfinite(timestamps).all():
        raise ValueError(f"{label} timestamps must be a non-empty finite vector")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{label} timestamps are not strictly increasing")


def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source_timestamps, target_timestamps, side="left")
    right = np.clip(right, 0, len(source_timestamps) - 1)
    left = np.clip(right - 1, 0, len(source_timestamps) - 1)
    choose_left = np.abs(target_timestamps - source_timestamps[left]) <= np.abs(
        source_timestamps[right] - target_timestamps
    )
    indices = np.where(choose_left, left, right)
    errors = np.abs(source_timestamps[indices] - target_timestamps)
    return indices.astype(np.int64), errors


def interpolation_valid(source_timestamps: np.ndarray, target_timestamps: np.ndarray, max_gap_ms: float) -> np.ndarray:
    insertion = np.searchsorted(source_timestamps, target_timestamps, side="left")
    exact_index = np.clip(insertion, 0, len(source_timestamps) - 1)
    exact = source_timestamps[exact_index] == target_timestamps
    left = np.clip(insertion - 1, 0, len(source_timestamps) - 1)
    right = np.clip(insertion, 0, len(source_timestamps) - 1)
    inside = (target_timestamps >= source_timestamps[0]) & (target_timestamps <= source_timestamps[-1])
    gaps = source_timestamps[right] - source_timestamps[left]
    return inside & (exact | (gaps <= max_gap_ms))


def interpolate_rows(source_timestamps: np.ndarray, values: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(target_timestamps, source_timestamps, values)
    return np.stack(
        [np.interp(target_timestamps, source_timestamps, values[:, column]) for column in range(values.shape[1])],
        axis=-1,
    )


def fixed_rate_timestamps(start_ms: float, end_ms: float, fps: int = FPS) -> np.ndarray:
    if end_ms < start_ms:
        return np.empty((0,), dtype=np.float64)
    count = int(math.floor((end_ms - start_ms) * fps / 1000.0 + 1e-9)) + 1
    timestamps = start_ms + np.arange(count, dtype=np.float64) * (1000.0 / fps)
    return np.minimum(timestamps, end_ms)


def contiguous_true_runs(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2) if end - start >= min_length]


def close_fraction(width_mm: np.ndarray | float) -> np.ndarray:
    width = np.asarray(width_mm, dtype=np.float64)
    return 1.0 - np.clip(width, 0.0, GRIPPER_MAX_WIDTH_MM) / GRIPPER_MAX_WIDTH_MM


def deterministic_split(group_key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios require train>0, val>=0, and train+val<1")
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < train_ratio:
        return "train"
    if unit < train_ratio + val_ratio:
        return "val"
    return "test"


def load_task_descriptions(path: Path) -> dict[str, dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Task description JSON must be an object: {path}")
    result: dict[str, dict[str, str]] = {}
    for task_id, row in value.items():
        if not isinstance(row, dict):
            continue
        english = str(row.get("task_description_english", "")).strip()
        chinese = str(row.get("task_description_chinese", "")).strip()
        if english:
            result[str(task_id)] = {"english": english, "chinese": chinese}
    return result


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=json_ready) + "\n")


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path, "".join(json.dumps(record, ensure_ascii=False, default=json_ready) + "\n" for record in records)
    )


__all__ = [
    "ACTION_LAYOUT",
    "CHUNK_LENGTH",
    "DEFAULT_DERIVED_ROOT",
    "DEFAULT_SOURCE_ROOT",
    "DEFAULT_TASK_DESCRIPTIONS",
    "EXTERIOR_CAMERAS",
    "FPS",
    "GRIPPER_MAX_WIDTH_MM",
    "PRIMARY_SERIAL",
    "EpisodeIdentity",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "close_fraction",
    "contiguous_true_runs",
    "deterministic_split",
    "fixed_rate_timestamps",
    "interpolate_rows",
    "interpolation_valid",
    "load_color_timestamps",
    "load_gripper_width_stream",
    "load_joint_stream",
    "load_object_npy",
    "load_task_descriptions",
    "load_tcp_base_stream",
    "nearest_indices",
    "parse_episode_name",
    "sha256_file",
]
