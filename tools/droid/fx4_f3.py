# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""F3 split, motion-aware window starts, and distribution control."""

from __future__ import annotations

import collections
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.droid.fx4_f2 import F2Config, Trajectory, load_accepted_trajectories, load_episode_metadata
from tools.droid.fx4_io import json_fingerprint, load_jsonl, sha256_file, write_checksums, write_json, write_jsonl

WINDOW_START_RULE = "motion_contact_window_start_v1"


@dataclass(frozen=True)
class F3Config:
    seed: int = 42
    val_ratio: float = 0.03
    translation_m: float = 0.005
    rotation_deg: float = 3.0
    gripper_crossing: float = 0.5
    gripper_change: float = 0.1
    max_gap_frames: int = 12
    gap_probe_frames: tuple[int, ...] = (8, 12, 16)
    max_windows_per_episode: int = 64
    max_windows_per_normalized_task: int = 4096


def _range_valid_start_bounds(range_record: dict[str, Any], length: int, chunk_length: int) -> tuple[int, int]:
    start = max(0, int(range_record["source_start"]))
    stop = min(int(range_record["source_end"]) - chunk_length, length - chunk_length)
    if stop <= start:
        return start, start
    return start, stop


def candidate_window_starts(
    trajectory: Trajectory,
    *,
    start: int,
    stop: int,
    config: F3Config,
    max_gap_frames: int,
) -> list[dict[str, Any]]:
    """Select starts on the original timeline using the event union."""
    if stop <= start:
        return []
    smooth_xyz = trajectory.smooth_xyz
    smooth_rotation = trajectory.smooth_rotation
    gripper = trajectory.gripper_close_fraction
    selected = [{"start": start, "events": ["range_start"]}]
    accepted = start
    accumulated_translation = 0.0
    accumulated_rotation = 0.0
    for index in range(start + 1, stop):
        accumulated_translation += float(np.linalg.norm(smooth_xyz[index] - smooth_xyz[index - 1]))
        accumulated_rotation += math.degrees((smooth_rotation[index - 1].inv() * smooth_rotation[index]).magnitude())
        events: list[str] = []
        if accumulated_translation >= config.translation_m:
            events.append("translation")
        if accumulated_rotation >= config.rotation_deg:
            events.append("rotation")
        crossed = (gripper[accepted] < config.gripper_crossing <= gripper[index]) or (
            gripper[accepted] >= config.gripper_crossing > gripper[index]
        )
        if crossed:
            events.append("gripper_crossing")
        elif abs(float(gripper[index] - gripper[accepted])) >= config.gripper_change:
            events.append("gripper_change")
        if index - accepted >= max_gap_frames:
            events.append("max_gap")
        if events:
            selected.append({"start": index, "events": events})
            accepted = index
            accumulated_translation = 0.0
            accumulated_rotation = 0.0
    last = stop - 1
    if selected[-1]["start"] != last:
        selected.append({"start": last, "events": ["range_end"]})
    elif "range_end" not in selected[-1]["events"]:
        selected[-1]["events"].append("range_end")
    return selected


def cap_episode_candidates(candidates_by_range: list[list[dict[str, Any]]], limit: int) -> set[tuple[int, int]]:
    flat = [(range_index, item) for range_index, items in enumerate(candidates_by_range) for item in items]
    if len(flat) <= limit:
        return {(range_index, int(item["start"])) for range_index, item in flat}
    mandatory = {
        (range_index, int(items[0]["start"])) for range_index, items in enumerate(candidates_by_range) if items
    } | {(range_index, int(items[-1]["start"])) for range_index, items in enumerate(candidates_by_range) if items}
    if len(mandatory) > limit:
        raise ValueError(f"Episode has {len(mandatory)} mandatory range endpoints, above cap {limit}")
    optional = [
        (range_index, int(item["start"]))
        for range_index, item in flat
        if (range_index, int(item["start"])) not in mandatory
    ]
    remaining = limit - len(mandatory)
    if remaining:
        indices = np.linspace(0, len(optional) - 1, remaining, dtype=int)
        mandatory.update(optional[index] for index in indices)
    return mandatory


def scene_group_id(episode_id: str) -> str:
    parts = episode_id.split("/")
    institution = parts[0] if parts else "unknown"
    date = parts[2] if len(parts) > 2 else "unknown_date"
    return f"{institution}/{date}"


def assign_group_splits(records: list[dict[str, Any]], config: F3Config) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    split_by_group = {
        group: (
            "val"
            if int.from_bytes(hashlib.sha256(f"{config.seed}:{group}".encode()).digest()[:8], "big") / 2**64
            < config.val_ratio
            else "train"
        )
        for group in groups
    }
    families = sorted({record["task_family"] for record in records})
    for family in families:
        if any(split_by_group[record["group_id"]] == "val" and record["task_family"] == family for record in records):
            continue
        candidates = sorted(
            group
            for group, group_records in groups.items()
            if split_by_group[group] == "train" and any(record["task_family"] == family for record in group_records)
        )
        if not candidates:
            raise ValueError(f"Cannot create validation coverage for family {family}")
        split_by_group[candidates[0]] = "val"
    return split_by_group


def _stable_window_score(seed: int, task: str, episode_index: int, start: int) -> bytes:
    return hashlib.sha256(f"{seed}:{task}:{episode_index}:{start}".encode()).digest()


def apply_task_cap(records: list[dict[str, Any]], config: F3Config) -> None:
    windows_by_task: dict[str, list[tuple[bytes, dict[str, Any], dict[str, Any], int]]] = collections.defaultdict(list)
    for record in records:
        if record["split"] != "train":
            continue
        for range_record in record["kept_ranges"]:
            for start in range_record["selected_window_starts_c32"]:
                windows_by_task[record["normalized_task"]].append(
                    (
                        _stable_window_score(
                            config.seed, record["normalized_task"], record["episode_index"], int(start)
                        ),
                        record,
                        range_record,
                        int(start),
                    )
                )
    for windows in windows_by_task.values():
        if len(windows) <= config.max_windows_per_normalized_task:
            continue
        keep = {
            (record["episode_index"], range_record["range_index"], start)
            for _, record, range_record, start in sorted(windows, key=lambda item: item[0])[
                : config.max_windows_per_normalized_task
            ]
        }
        for _, record, range_record, start in windows:
            key = (record["episode_index"], range_record["range_index"], start)
            if key not in keep:
                range_record["selected_window_starts_c32"].remove(start)


def _continuous_window_gate(trajectory: Trajectory, starts: list[int], chunk_length: int, fps: float) -> bool:
    for start in starts:
        timestamps = trajectory.timestamp[start : start + chunk_length + 1]
        if len(timestamps) != chunk_length + 1:
            return False
        if np.max(np.abs(np.diff(timestamps) - 1.0 / fps), initial=0.0) > 2e-4:
            return False
    return True


def build_f3(
    *, success_root: Path, f2_dir: Path, config: F3Config
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    f2_path = f2_dir / "f2_episodes.jsonl"
    f2_summary_path = f2_dir / "f2_summary.json"
    f2_records = {
        int(record["episode_index"]): record for record in load_jsonl(f2_path) if record["f2_status"] == "accepted"
    }
    metadata = load_episode_metadata(success_root)
    trajectories = load_accepted_trajectories(success_root, f2_records, metadata, F2Config())

    records: list[dict[str, Any]] = []
    gap_probe_counts: collections.Counter[int] = collections.Counter()
    event_counts: collections.Counter[str] = collections.Counter()
    waypoint_retained_frames = 0
    waypoint_source_frames = 0
    waypoint_complete_windows = 0
    waypoint_chunk_durations: list[float] = []
    for episode_index, f2_record in sorted(f2_records.items()):
        trajectory = trajectories[episode_index]
        accepted_ranges = [item for item in f2_record["ranges"] if item.get("f2_status") == "accepted"]
        main_candidates: list[list[dict[str, Any]]] = []
        c16_candidates: list[list[dict[str, Any]]] = []
        for range_record in accepted_ranges:
            start32, stop32 = _range_valid_start_bounds(range_record, len(trajectory.timestamp), 32)
            candidates32 = candidate_window_starts(
                trajectory, start=start32, stop=stop32, config=config, max_gap_frames=config.max_gap_frames
            )
            main_candidates.append(candidates32)
            start16, stop16 = _range_valid_start_bounds(range_record, len(trajectory.timestamp), 16)
            c16_candidates.append(
                candidate_window_starts(
                    trajectory, start=start16, stop=stop16, config=config, max_gap_frames=config.max_gap_frames
                )
            )
            for gap in config.gap_probe_frames:
                gap_probe_counts[gap] += len(
                    candidate_window_starts(trajectory, start=start32, stop=stop32, config=config, max_gap_frames=gap)
                )
            waypoint_candidates = candidate_window_starts(
                trajectory, start=start32, stop=stop32, config=config, max_gap_frames=10**9
            )
            waypoint_indices = [int(item["start"]) for item in waypoint_candidates]
            waypoint_retained_frames += len(waypoint_indices)
            waypoint_source_frames += max(0, stop32 - start32)
            waypoint_complete_windows += max(0, len(waypoint_indices) - 32)
            waypoint_chunk_durations.extend(
                float(trajectory.timestamp[waypoint_indices[index + 32]] - trajectory.timestamp[start])
                for index, start in enumerate(waypoint_indices[:-32])
            )
        selected = cap_episode_candidates(main_candidates, config.max_windows_per_episode)
        selected16 = cap_episode_candidates(c16_candidates, config.max_windows_per_episode)
        kept_ranges: list[dict[str, Any]] = []
        for range_index, range_record in enumerate(accepted_ranges):
            starts32 = [
                int(item["start"])
                for item in main_candidates[range_index]
                if (range_index, int(item["start"])) in selected
            ]
            starts16 = [
                int(item["start"])
                for item in c16_candidates[range_index]
                if (range_index, int(item["start"])) in selected16
            ]
            for item in main_candidates[range_index]:
                if (range_index, int(item["start"])) in selected:
                    event_counts.update(item["events"])
            kept_ranges.append(
                {
                    "range_index": range_index,
                    "source_start": range_record["source_start"],
                    "source_end": range_record["source_end"],
                    "selected_window_starts_c32": starts32,
                    "selected_window_starts_c16": starts16,
                }
            )
        group_id = scene_group_id(f2_record["episode_id"])
        records.append(
            {
                "episode_index": episode_index,
                "episode_id": f2_record["episode_id"],
                "task_family": f2_record["task_family"],
                "tasks_raw": f2_record["tasks_raw"],
                "normalized_task": f2_record["tasks_annotations"][0],
                "group_id": group_id,
                "kept_ranges": kept_ranges,
                "pose_smoothing_rule": f2_record["pose_smoothing_rule"],
                "window_start_rule": WINDOW_START_RULE,
                "force_event_source": "unavailable",
                "source_revision": "5c11a20accb11497270a5247a7f1e66ad04c956c",
                "reason_codes": [],
            }
        )

    split_by_group = assign_group_splits(records, config)
    for record in records:
        record["split"] = split_by_group[record["group_id"]]
    apply_task_cap(records, config)
    records = [
        record
        for record in records
        if record["split"] == "val" or any(item["selected_window_starts_c32"] for item in record["kept_ranges"])
    ]

    all_families = {record["task_family"] for record in records}
    family_counts = collections.Counter(record["task_family"] for record in records if record["split"] == "train")
    largest_family = max(family_counts.values(), default=1)
    for record in records:
        record["sample_weight"] = (
            min(3.0, math.sqrt(largest_family / family_counts[record["task_family"]]))
            if record["split"] == "train"
            else 1.0
        )

    continuity_failures = 0
    for record in records:
        starts = [start for item in record["kept_ranges"] for start in item["selected_window_starts_c32"]]
        if not _continuous_window_gate(trajectories[record["episode_index"]], starts, 32, 15.0):
            continuity_failures += 1

    train = [record for record in records if record["split"] == "train"]
    val = [record for record in records if record["split"] == "val"]
    train_groups = {record["group_id"] for record in train}
    val_groups = {record["group_id"] for record in val}
    train_episodes = {record["episode_index"] for record in train}
    val_episodes = {record["episode_index"] for record in val}
    window_counts = {
        split: sum(
            len(range_record["selected_window_starts_c32"])
            for record in split_records
            for range_record in record["kept_ranges"]
        )
        for split, split_records in (("train", train), ("val", val))
    }
    family_split_counts = {
        split: dict(sorted(collections.Counter(record["task_family"] for record in split_records).items()))
        for split, split_records in (("train", train), ("val", val))
    }
    inputs = {
        "f2_summary": {"path": str(f2_summary_path), "sha256": sha256_file(f2_summary_path)},
        "f2_episodes": {"path": str(f2_path), "sha256": sha256_file(f2_path)},
    }
    gates = {
        "episode_overlap": len(train_episodes & val_episodes),
        "group_overlap": len(train_groups & val_groups),
        "continuous_15hz_failures": continuity_failures,
        "all_families_in_train": all_families <= set(family_split_counts["train"]),
        "all_families_in_val": all_families <= set(family_split_counts["val"]),
        "val_ratio_within_one_percent": abs(len(val) / len(records) - config.val_ratio) <= 0.01,
    }
    waypoint_duration_array = np.asarray(waypoint_chunk_durations, dtype=np.float64)
    summary = {
        "stage": "f3",
        "config": {**config.__dict__, "gap_probe_frames": list(config.gap_probe_frames)},
        "counts": {
            "episodes": {"train": len(train), "val": len(val)},
            "windows_c32": window_counts,
            "families": family_split_counts,
            "gap_probe_candidate_starts": {str(key): value for key, value in sorted(gap_probe_counts.items())},
            "selected_event_reasons": dict(sorted(event_counts.items())),
            "waypoint_policy_ablation": {
                "retained_frames": waypoint_retained_frames,
                "source_candidate_frames": waypoint_source_frames,
                "retained_frame_ratio": waypoint_retained_frames / waypoint_source_frames,
                "complete_windows_c32": waypoint_complete_windows,
                "chunk_duration_s": {
                    "p05": float(np.quantile(waypoint_duration_array, 0.05)) if waypoint_duration_array.size else 0.0,
                    "median": float(np.quantile(waypoint_duration_array, 0.5)) if waypoint_duration_array.size else 0.0,
                    "p95": float(np.quantile(waypoint_duration_array, 0.95)) if waypoint_duration_array.size else 0.0,
                },
                "note": "variable-delta-t probe only; not used by the fixed-15Hz manifest",
            },
        },
        "gates": gates,
        "inputs": inputs,
    }
    summary["input_fingerprint"] = json_fingerprint({"inputs": inputs, "config": summary["config"]})
    return summary, train, val


def write_f3(output_dir: Path, summary: dict[str, Any], train: list[dict[str, Any]], val: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "f3_summary.json"
    train_path = output_dir / "manifest_train.jsonl"
    val_path = output_dir / "manifest_val.jsonl"
    write_json(summary_path, summary)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_checksums(output_dir / "F3_SHA256SUMS", (summary_path, train_path, val_path))
