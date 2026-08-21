#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compute train-only 10D EEF action statistics for DROID Fx4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.droid.build_fx4_manifest import resolve_success_root
from tools.droid.fx4_f2 import F2Config, Trajectory, load_accepted_trajectories, load_episode_metadata
from tools.droid.fx4_io import (
    json_fingerprint,
    load_jsonl,
    refuse_source_output,
    sha256_file,
    write_checksums,
    write_json,
)


def window_actions(trajectory: Trajectory, start: int, chunk_length: int = 32) -> np.ndarray:
    """Return inv(T_start) @ T_future plus future open_fraction."""
    stop = start + chunk_length + 1
    if start < 0 or stop > len(trajectory.timestamp):
        raise ValueError(f"Window [{start}, {stop}) exceeds trajectory length {len(trajectory.timestamp)}")
    base_rotation = trajectory.smooth_rotation[start]
    future_rotation = trajectory.smooth_rotation[start + 1 : stop]
    relative_rotation = base_rotation.inv() * future_rotation
    relative_xyz = base_rotation.inv().apply(trajectory.smooth_xyz[start + 1 : stop] - trajectory.smooth_xyz[start])
    matrices = relative_rotation.as_matrix()
    rot6d = np.concatenate((matrices[:, :, 0], matrices[:, :, 1]), axis=1)
    open_fraction = 1.0 - trajectory.gripper_close_fraction[start + 1 : stop]
    action = np.concatenate((relative_xyz, rot6d, open_fraction[:, None]), axis=1).astype(np.float32)
    if action.shape != (chunk_length, 10) or not np.isfinite(action).all():
        raise ValueError(f"Invalid EEF action at start {start}: shape={action.shape}")
    return action


def action_stats(actions: np.ndarray) -> dict[str, list[float] | int]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
        raise ValueError("actions must be finite [N,10]")
    return {
        "count": int(len(actions)),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.quantile(actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(actions, 0.99, axis=0).tolist(),
    }


def build_stats(*, success_root: Path, manifest_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_dir / "manifest_train.jsonl"
    records = load_jsonl(manifest_path)
    record_map = {int(record["episode_index"]): record for record in records}
    metadata = load_episode_metadata(success_root)
    trajectories = load_accepted_trajectories(success_root, record_map, metadata, F2Config())
    chunks: list[np.ndarray] = []
    expected_windows = 0
    for record_number, record in enumerate(records, start=1):
        trajectory = trajectories[int(record["episode_index"])]
        episode_actions: list[np.ndarray] = []
        for range_record in record["kept_ranges"]:
            starts = [int(value) for value in range_record["selected_window_starts_c32"]]
            expected_windows += len(starts)
            episode_actions.extend(window_actions(trajectory, start) for start in starts)
        if episode_actions:
            chunks.append(np.concatenate(episode_actions, axis=0))
        if record_number % 1000 == 0:
            print(f"Action stats progress: {record_number}/{len(records)} episodes", flush=True)
    actions = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 10), dtype=np.float32)
    stats = action_stats(actions)
    if stats["count"] != expected_windows * 32:
        raise ValueError("Action statistics count does not match manifest windows × chunk length")
    block = {key: value for key, value in stats.items() if key != "count"}
    result = {
        "metadata": {
            "dataset": "Cosmos3-DROID/success Fx4",
            "split": "train",
            "action_contract": "inv(T_current)@T_future + future open_fraction",
            "action_dim": 10,
            "chunk_length": 32,
            "rotation_format": "rot6d_columns_0_1",
            "pose_smoothing_rule": "se3_local_geodesic_binomial_w5_endpoints_fixed_v1",
            "normalization": "quantile_rot",
            "apply_forward_clamp": False,
            "num_windows": expected_windows,
            "num_action_rows": stats["count"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "global": block,
        "global_raw": block,
    }
    result["input_fingerprint"] = json_fingerprint(result["metadata"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    success_root = resolve_success_root(args.dataset_root)
    output = args.output.expanduser().resolve()
    refuse_source_output(success_root, output.parent)
    if output.exists():
        raise FileExistsError(output)
    result = build_stats(success_root=success_root, manifest_dir=args.manifest_dir.expanduser().resolve())
    write_json(output, result)
    write_checksums(output.with_name("ACTION_STATS_SHA256SUMS"), (output,))
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
