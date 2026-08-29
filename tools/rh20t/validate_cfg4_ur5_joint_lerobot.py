# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validate RH20T cfg4 UR5-joint LeRobot structure, values, tasks, and videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from tools.rh20t.convert_cfg4_ur5_joint_to_lerobot import _resample_segment
from tools.rh20t.rh20t_io import ACTION_LAYOUT, EXTERIOR_CAMERAS, FPS, atomic_write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_table(dataset: LeRobotDataset):
    episodes = dataset.meta.episodes
    return episodes.to_pandas() if hasattr(episodes, "to_pandas") else episodes


def _episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    episodes = _episode_table(dataset)
    columns = set(episodes.columns)
    row = episodes.loc[episodes["episode_index"] == episode_index].iloc[0]
    if {"dataset_from_index", "dataset_to_index"} <= columns:
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])
    values = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    indices = np.flatnonzero(values == episode_index)
    if not len(indices):
        raise ValueError(f"No frames found for episode_index={episode_index}")
    return int(indices[0]), int(indices[-1]) + 1


def _video_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _video_chunk_expectations(
    meta: LeRobotDatasetMetadata,
    root: Path,
    episode_rows,
    episode_lengths: dict[int, int],
) -> list[dict[str, Any]]:
    """Group episode ranges by their physical LeRobot v3 video chunk."""
    chunks: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, row in episode_rows.iterrows():
        episode_index = int(row["episode_index"]) if "episode_index" in episode_rows.columns else int(row_index)
        for key in ("observation.images.exterior_1", "observation.images.exterior_2"):
            path = meta.get_video_file_path(episode_index, key)
            if not path.is_absolute():
                path = root / path
            identity = (key, str(path))
            chunk = chunks.setdefault(
                identity,
                {
                    "key": key,
                    "path": str(path),
                    "episode_indices": [],
                    "expected_frames": 0,
                },
            )
            chunk["episode_indices"].append(episode_index)
            chunk["expected_frames"] += episode_lengths[episode_index]
    return list(chunks.values())


def validate(
    root: Path,
    *,
    manifest: Path | None = None,
    expected_limit: int | None = None,
    video_episode_count: int = 3,
) -> dict[str, Any]:
    meta = LeRobotDatasetMetadata(repo_id="local", root=root, revision="local")
    dataset = LeRobotDataset(repo_id="local", root=root, revision="local")
    features = meta.info.get("features", {})
    required = {
        "action",
        "observation.state.joint",
        "observation.state.gripper",
        "observation.images.exterior_1",
        "observation.images.exterior_2",
        "source.timestamp_ms",
        "source.camera_alignment_error_ms",
    }
    errors: list[str] = []
    if float(meta.fps) != FPS:
        errors.append(f"fps={meta.fps}, expected {FPS}")
    missing = sorted(required - set(features))
    if missing:
        errors.append(f"missing features: {missing}")
    action_feature = features.get("action", {})
    if tuple(action_feature.get("shape", ())) != (7,) or action_feature.get("dtype") != "float32":
        errors.append(f"invalid action feature: {action_feature}")
    names = action_feature.get("names")
    layout = tuple(names.get("motors", ())) if isinstance(names, dict) else tuple(names or ())
    if layout != ACTION_LAYOUT:
        errors.append(f"action layout={layout!r}, expected {ACTION_LAYOUT!r}")

    manifest_records = _read_jsonl(manifest) if manifest is not None else []
    expected_records = manifest_records[:expected_limit] if expected_limit is not None else manifest_records
    if expected_records and int(meta.total_episodes) != len(expected_records):
        errors.append(f"episodes={meta.total_episodes}, expected {len(expected_records)} from manifest")
    expected_by_episode: dict[int, dict[str, Any]] = {}
    if expected_records:
        expected_by_segment = {str(record["segment_id"]): record for record in expected_records}
        ledger = root / "conversion" / "ledger.jsonl"
        if ledger.is_file():
            for record in _read_jsonl(ledger):
                expected = expected_by_segment.get(str(record.get("segment_id")))
                if expected is not None and record.get("status") == "saved":
                    expected_by_episode[int(record["lerobot_episode_index"])] = expected
        else:
            expected_by_episode = dict(enumerate(expected_records))

    episode_rows = _episode_table(dataset).sort_values("episode_index")
    numeric_summary: list[dict[str, Any]] = []
    episode_lengths: dict[int, int] = {}
    for row_index, row in episode_rows.iterrows():
        episode_index = int(row["episode_index"]) if "episode_index" in episode_rows.columns else int(row_index)
        start, end = _episode_bounds(dataset, episode_index)
        episode_lengths[episode_index] = end - start
        sample = dataset.hf_dataset.select(range(start, end))
        action = np.asarray(sample["action"], dtype=np.float32)
        joint = np.asarray(sample["observation.state.joint"], dtype=np.float32)
        gripper = np.asarray(sample["observation.state.gripper"], dtype=np.float32).reshape(-1)
        source_timestamp = np.asarray(sample["source.timestamp_ms"], dtype=np.float64).reshape(-1)
        alignment = np.asarray(sample["source.camera_alignment_error_ms"], dtype=np.float32)
        if action.shape != (end - start, 7):
            errors.append(f"episode {episode_index}: action shape {action.shape}")
        if not np.isfinite(action).all() or not np.isfinite(source_timestamp).all():
            errors.append(f"episode {episode_index}: non-finite numeric values")
        if len(source_timestamp) > 1 and np.any(np.diff(source_timestamp) <= 0):
            errors.append(f"episode {episode_index}: source timestamps not strictly increasing")
        if np.max(np.abs(joint), initial=0.0) > 2 * np.pi + 0.25:
            errors.append(f"episode {episode_index}: joint magnitude exceeds UR5 gate")
        if np.any((gripper < 0) | (gripper > 1)):
            errors.append(f"episode {episode_index}: gripper outside [0,1]")
        if not np.allclose(action[:, :6], joint) or not np.allclose(action[:, 6], gripper):
            errors.append(f"episode {episode_index}: storage-time action/state mismatch")
        if np.max(alignment, initial=0.0) > 100.0 + 1e-3:
            errors.append(f"episode {episode_index}: camera alignment exceeds 100 ms")
        expected_record = expected_by_episode.get(episode_index)
        if expected_record is not None:
            expected = _resample_segment(expected_record)
            expected_action = np.concatenate(
                (expected["joint"], expected["gripper"][:, None]),
                axis=1,
            )
            expected_timestamp = np.rint(expected["target_timestamps"]).astype(np.int64)
            expected_alignment = np.stack(
                [expected["camera_errors"][serial] for serial in EXTERIOR_CAMERAS],
                axis=1,
            )
            if action.shape != expected_action.shape or not np.allclose(action, expected_action, atol=1e-6, rtol=0):
                errors.append(f"episode {episode_index}: action differs from manifest source resampling")
            if not np.array_equal(source_timestamp.astype(np.int64), expected_timestamp):
                errors.append(f"episode {episode_index}: source timestamps differ from manifest source resampling")
            if alignment.shape != expected_alignment.shape or not np.allclose(
                alignment, expected_alignment, atol=1e-5, rtol=0
            ):
                errors.append(f"episode {episode_index}: camera alignment differs from manifest source resampling")
            tasks = {str(task) for task in np.asarray(row["tasks"]).reshape(-1)}
            if str(expected_record["task_instruction_en"]) not in tasks:
                errors.append(f"episode {episode_index}: task text differs from manifest")
        numeric_summary.append(
            {
                "episode_index": episode_index,
                "frames": end - start,
                "source_duration_s": float((source_timestamp[-1] - source_timestamp[0]) / 1000),
                "max_camera_alignment_error_ms": float(np.max(alignment, initial=0.0)),
            }
        )

    selected_episodes = (
        set(episode_lengths)
        if video_episode_count < 0
        else set(episode_lengths) & set(episode_rows.head(video_episode_count)["episode_index"].astype(int))
    )
    video_checks: list[dict[str, Any]] = []
    for chunk in _video_chunk_expectations(meta, root, episode_rows, episode_lengths):
        if not selected_episodes.intersection(chunk["episode_indices"]):
            continue
        count = _video_frame_count(Path(chunk["path"]))
        chunk["frames"] = count
        video_checks.append(chunk)
        if count != chunk["expected_frames"]:
            errors.append(
                f"{chunk['key']} video chunk {chunk['path']} frames={count}, parquet frames={chunk['expected_frames']}"
            )

    report = {
        "status": "pass" if not errors else "fail",
        "root": str(root),
        "fps": float(meta.fps),
        "episodes": int(meta.total_episodes),
        "frames": int(meta.total_frames),
        "features": sorted(features),
        "numeric_summary": numeric_summary,
        "source_resampling_checks": len(expected_by_episode),
        "video_checks": video_checks,
        "errors": errors,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-limit", type=int)
    parser.add_argument("--video-episode-count", type=int, default=3)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        args.root,
        manifest=args.manifest,
        expected_limit=args.expected_limit,
        video_episode_count=args.video_episode_count,
    )
    report_path = args.report or args.root / "validation_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
