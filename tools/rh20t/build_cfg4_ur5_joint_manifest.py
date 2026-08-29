# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build fixed-15Hz RH20T cfg4 manifests without modifying source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.rh20t.rh20t_io import (
    ACTION_LAYOUT,
    CHUNK_LENGTH,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_TASK_DESCRIPTIONS,
    EXTERIOR_CAMERAS,
    FPS,
    GRIPPER_MAX_WIDTH_MM,
    PRIMARY_SERIAL,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    close_fraction,
    contiguous_true_runs,
    deterministic_split,
    fixed_rate_timestamps,
    interpolate_rows,
    interpolation_valid,
    load_color_timestamps,
    load_gripper_width_stream,
    load_joint_stream,
    load_task_descriptions,
    nearest_indices,
    parse_episode_name,
    sha256_file,
)

DEFAULT_TASK_POLICY = Path(__file__).with_name("task_policy_cfg4_ur5_v1.json")


@dataclass(frozen=True, slots=True)
class BuildConfig:
    fps: int = FPS
    chunk_length: int = CHUNK_LENGTH
    min_rating: int = 2
    accepted_calib_quality: tuple[int, ...] = (1, 2, 3)
    min_scene: int = 1
    max_scene: int = 10
    max_camera_error_ms: float = 100.0
    max_interpolation_gap_ms: float = 250.0
    split_seed: int = 42
    train_ratio: float = 0.9
    val_ratio: float = 0.05


class EpisodeRejected(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def load_task_policy(
    path: Path,
    observed_task_ids: set[str],
) -> tuple[dict[str, Any], set[str], dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not str(value.get("policy_id", "")).strip():
        raise ValueError(f"Task policy must contain a non-empty policy_id: {path}")
    included_values = value.get("included_task_ids")
    excluded_values = value.get("excluded_tasks")
    if not isinstance(included_values, list) or not isinstance(excluded_values, dict):
        raise ValueError(f"Task policy must contain included_task_ids and excluded_tasks: {path}")
    included = {str(task_id) for task_id in included_values}
    excluded = {str(task_id): str(reason) for task_id, reason in excluded_values.items()}
    if len(included) != len(included_values):
        raise ValueError(f"Task policy contains duplicate included_task_ids: {path}")
    if included & set(excluded):
        raise ValueError(f"Task policy includes and excludes the same task IDs: {path}")
    reviewed = included | set(excluded)
    if reviewed != observed_task_ids:
        missing = sorted(observed_task_ids - reviewed)
        unknown = sorted(reviewed - observed_task_ids)
        raise ValueError(f"Task policy/catalog mismatch: missing={missing}, unknown={unknown}")
    if any(not reason.strip() for reason in excluded.values()):
        raise ValueError(f"Task policy exclusion reasons must not be empty: {path}")
    return value, included, excluded


def _stream_summary(timestamps: np.ndarray) -> dict[str, Any]:
    gaps = np.diff(timestamps)
    return {
        "count": int(len(timestamps)),
        "first_ms": float(timestamps[0]),
        "last_ms": float(timestamps[-1]),
        "median_gap_ms": float(np.median(gaps)) if len(gaps) else None,
        "max_gap_ms": float(np.max(gaps)) if len(gaps) else None,
    }


def _quality_gate(identity, metadata: dict[str, Any], config: BuildConfig) -> None:
    if identity.cfg != 4:
        raise EpisodeRejected("wrong_configuration", f"cfg={identity.cfg}")
    if not config.min_scene <= identity.scene_number <= config.max_scene:
        raise EpisodeRejected("scene_out_of_scope", f"scene={identity.scene_number}")
    rating = metadata.get("rating")
    if not isinstance(rating, int) or rating < config.min_rating:
        raise EpisodeRejected("low_or_missing_rating", f"rating={rating!r}")
    calib_quality = metadata.get("calib_quality")
    if calib_quality not in config.accepted_calib_quality:
        raise EpisodeRejected("calibration_quality", f"calib_quality={calib_quality!r}")
    if not isinstance(metadata.get("finish_time"), int):
        raise EpisodeRejected("missing_finish_time", f"finish_time={metadata.get('finish_time')!r}")


def _inspect_robot_episode(
    episode_path: Path,
    task: dict[str, str],
    config: BuildConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = parse_episode_name(episode_path.name)
    try:
        metadata = json.loads((episode_path / "metadata.json").read_text(encoding="utf-8"))
    except Exception as error:
        raise EpisodeRejected("metadata_unreadable", str(error)) from error
    _quality_gate(identity, metadata, config)
    finish_time_ms = float(metadata["finish_time"])

    transformed = episode_path / "transformed"
    try:
        joint_ts, joints = load_joint_stream(transformed / "joint.npy", PRIMARY_SERIAL, finish_time_ms)
        gripper_ts, width_mm = load_gripper_width_stream(transformed / "gripper.npy", PRIMARY_SERIAL, finish_time_ms)
    except (FileNotFoundError, ValueError) as error:
        raise EpisodeRejected("robot_stream_invalid", str(error)) from error

    camera_timestamps: dict[str, np.ndarray] = {}
    camera_files: dict[str, Path] = {}
    for serial in EXTERIOR_CAMERAS:
        camera_dir = episode_path / f"cam_{serial}"
        video = camera_dir / "color.mp4"
        if not video.is_file() or video.stat().st_size <= 0:
            raise EpisodeRejected("camera_video_missing", str(video))
        try:
            camera_timestamps[serial] = load_color_timestamps(camera_dir / "timestamps.npy", finish_time_ms)
        except (FileNotFoundError, ValueError) as error:
            raise EpisodeRejected("camera_timestamps_invalid", f"{serial}: {error}") from error
        camera_files[serial] = video

    start_ms = max(joint_ts[0], gripper_ts[0], *(value[0] for value in camera_timestamps.values()))
    end_ms = min(joint_ts[-1], gripper_ts[-1], *(value[-1] for value in camera_timestamps.values()))
    target_timestamps = fixed_rate_timestamps(start_ms, end_ms, config.fps)
    if len(target_timestamps) < config.chunk_length + 1:
        raise EpisodeRejected("episode_too_short", f"target_frames={len(target_timestamps)}")

    valid = interpolation_valid(joint_ts, target_timestamps, config.max_interpolation_gap_ms)
    valid &= interpolation_valid(gripper_ts, target_timestamps, config.max_interpolation_gap_ms)
    camera_indices: dict[str, np.ndarray] = {}
    camera_errors: dict[str, np.ndarray] = {}
    for serial, timestamps in camera_timestamps.items():
        indices, errors = nearest_indices(timestamps, target_timestamps)
        camera_indices[serial] = indices
        camera_errors[serial] = errors
        valid &= errors <= config.max_camera_error_ms

    runs = contiguous_true_runs(valid, config.chunk_length + 1)
    if not runs:
        raise EpisodeRejected(
            "no_valid_c32_segment",
            f"target_frames={len(target_timestamps)} valid_frames={int(valid.sum())}",
        )

    split = deterministic_split(
        identity.group_key,
        config.split_seed,
        config.train_ratio,
        config.val_ratio,
    )
    records: list[dict[str, Any]] = []
    resampled_joints = interpolate_rows(joint_ts, np.unwrap(joints, axis=0), target_timestamps)
    resampled_width = interpolate_rows(gripper_ts, width_mm, target_timestamps)
    close = close_fraction(resampled_width)

    for segment_index, (start, end) in enumerate(runs):
        frame_count = end - start
        segment_id = f"{episode_path.name}__segment_{segment_index:02d}"
        records.append(
            {
                "schema_version": "rh20t_cfg4_ur5_joint_v1",
                "segment_id": segment_id,
                "segment_index": segment_index,
                "source_episode_id": episode_path.name,
                "source_path": str(episode_path),
                "task_id": identity.task_id,
                "task_instruction_en": task["english"],
                "task_instruction_zh": task["chinese"],
                "user_id": identity.user_id,
                "scene_id": identity.scene_id,
                "split": split,
                "rating": metadata["rating"],
                "calib_quality": metadata["calib_quality"],
                "target_fps": config.fps,
                "target_start_ms": float(target_timestamps[start]),
                "target_step_ms": 1000.0 / config.fps,
                "frame_count": frame_count,
                "window_count_c32": frame_count - config.chunk_length,
                "action_layout": ACTION_LAYOUT,
                "gripper_semantics": "close_fraction=1-clip(command_width_mm,0,85)/85",
                "joint_target_source": "measured_joint_proxy",
                "cameras": {
                    "exterior_1": EXTERIOR_CAMERAS[0],
                    "exterior_2": EXTERIOR_CAMERAS[1],
                },
                "source_streams": {
                    "joint": _stream_summary(joint_ts),
                    "gripper": _stream_summary(gripper_ts),
                    **{
                        f"camera_{serial}": {
                            **_stream_summary(camera_timestamps[serial]),
                            "video_path": str(camera_files[serial]),
                            "video_bytes": camera_files[serial].stat().st_size,
                        }
                        for serial in EXTERIOR_CAMERAS
                    },
                },
                "alignment": {
                    serial: {
                        "max_error_ms": float(camera_errors[serial][start:end].max()),
                        "p90_error_ms": float(np.quantile(camera_errors[serial][start:end], 0.9)),
                    }
                    for serial in EXTERIOR_CAMERAS
                },
            }
        )

    movement: dict[str, int] = {"steps": 0, "hold": 0, "low_motion": 0, "moving": 0}
    for start, end in runs:
        delta_joint = np.max(np.abs(np.diff(resampled_joints[start:end], axis=0)), axis=1)
        delta_gripper = np.abs(np.diff(close[start:end]))
        hold = (delta_joint < 0.002) & (delta_gripper < 0.01)
        low = (delta_joint < 0.01) & (delta_gripper < 0.03)
        movement["steps"] += len(delta_joint)
        movement["hold"] += int(hold.sum())
        movement["low_motion"] += int((low & ~hold).sum())
        movement["moving"] += int((~low).sum())

    kept_indices = np.concatenate([np.arange(start, end) for start, end in runs])
    metrics = {
        "split": split,
        "joint_values": resampled_joints[kept_indices],
        "gripper_width_mm": resampled_width[kept_indices],
        "movement": movement,
        "target_frames": len(target_timestamps),
        "valid_frames": int(valid.sum()),
        "segment_count": len(records),
        "camera_files": {serial: True for serial in EXTERIOR_CAMERAS},
    }
    return records, metrics


def _input_fingerprint(
    source_root: Path,
    task_descriptions: Path,
    task_policy_path: Path | None,
    config: BuildConfig,
    names: list[str],
) -> str:
    payload = {
        "source_root": str(source_root.resolve()),
        "task_descriptions": str(task_descriptions.resolve()),
        "task_descriptions_sha256": sha256_file(task_descriptions),
        "task_policy_sha256": sha256_file(task_policy_path) if task_policy_path is not None else None,
        "config": asdict(config),
        "directory_names": names,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_manifest(
    *,
    source_root: Path,
    task_descriptions_path: Path,
    task_policy_path: Path | None = None,
    config: BuildConfig,
    limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    tasks = load_task_descriptions(task_descriptions_path)
    directories = sorted(path for path in source_root.iterdir() if path.is_dir())
    names = [path.name for path in directories]
    inventory: list[dict[str, Any]] = []
    candidates: list[Path] = []
    auxiliary_count = 0
    human_count = 0
    for path in directories:
        try:
            identity = parse_episode_name(path.name)
        except ValueError:
            auxiliary_count += 1
            inventory.append({"source_episode_id": path.name, "kind": "auxiliary"})
            continue
        kind = "human" if identity.is_human else "robot"
        inventory.append(
            {
                "source_episode_id": path.name,
                "kind": kind,
                "task_id": identity.task_id,
                "user_id": identity.user_id,
                "scene_id": identity.scene_id,
                "cfg": identity.cfg,
            }
        )
        if identity.is_human:
            human_count += 1
        else:
            candidates.append(path)
    observed_task_ids = {parse_episode_name(path.name).task_id for path in candidates}
    if task_policy_path is None:
        task_policy = {"policy_id": "unfiltered", "decision_note": "No task policy supplied."}
        included_tasks, excluded_tasks = observed_task_ids, {}
    else:
        task_policy, included_tasks, excluded_tasks = load_task_policy(task_policy_path, observed_task_ids)
    if limit is not None:
        candidates = candidates[:limit]

    accepted: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    metric_rows: list[dict[str, Any]] = []
    task_source_counts: Counter[str] = Counter()
    task_segment_counts: Counter[str] = Counter()

    for episode_path in candidates:
        identity = parse_episode_name(episode_path.name)
        task_source_counts[identity.task_id] += 1
        if identity.task_id not in included_tasks:
            detail = excluded_tasks[identity.task_id]
            rejected.append(
                {
                    "source_episode_id": episode_path.name,
                    "source_path": str(episode_path),
                    "task_id": identity.task_id,
                    "user_id": identity.user_id,
                    "scene_id": identity.scene_id,
                    "reason": "task_policy_excluded",
                    "detail": detail,
                }
            )
            reason_counts["task_policy_excluded"] += 1
            continue
        task = tasks.get(identity.task_id)
        if task is None:
            rejected.append(
                {
                    "source_episode_id": episode_path.name,
                    "source_path": str(episode_path),
                    "reason": "task_description_missing",
                    "detail": identity.task_id,
                }
            )
            reason_counts["task_description_missing"] += 1
            continue
        try:
            records, metrics = _inspect_robot_episode(episode_path, task, config)
        except EpisodeRejected as error:
            rejected.append(
                {
                    "source_episode_id": episode_path.name,
                    "source_path": str(episode_path),
                    "task_id": identity.task_id,
                    "user_id": identity.user_id,
                    "scene_id": identity.scene_id,
                    "reason": error.reason,
                    "detail": str(error),
                }
            )
            reason_counts[error.reason] += 1
            continue
        for record in records:
            record["task_policy_id"] = task_policy["policy_id"]
            record["task_policy_decision"] = "include"
            accepted[record["split"]].append(record)
            task_segment_counts[record["task_id"]] += 1
        metric_rows.append(metrics)

    all_records = [record for split in ("train", "val", "test") for record in accepted[split]]
    source_accepted = len({record["source_episode_id"] for record in all_records})
    split_sources = {
        split: len({record["source_episode_id"] for record in records}) for split, records in accepted.items()
    }
    movement = Counter()
    for row in metric_rows:
        movement.update(row["movement"])
    joints = np.concatenate([row["joint_values"] for row in metric_rows], axis=0) if metric_rows else np.empty((0, 6))
    widths = np.concatenate([row["gripper_width_mm"] for row in metric_rows], axis=0) if metric_rows else np.empty((0,))
    train_rows = [row for row in metric_rows if row["split"] == "train"]
    train_joints = (
        np.concatenate([row["joint_values"] for row in train_rows], axis=0) if train_rows else np.empty((0, 6))
    )
    train_widths = (
        np.concatenate([row["gripper_width_mm"] for row in train_rows], axis=0) if train_rows else np.empty((0,))
    )
    train_actions = (
        np.concatenate((train_joints, close_fraction(train_widths)[:, None]), axis=1)
        if train_joints.size
        else np.empty((0, 7))
    )
    summary = {
        "schema_version": "rh20t_cfg4_ur5_joint_v1",
        "input_fingerprint": _input_fingerprint(
            source_root,
            task_descriptions_path,
            task_policy_path,
            config,
            names,
        ),
        "source_root": str(source_root),
        "task_descriptions": {
            "path": str(task_descriptions_path),
            "sha256": sha256_file(task_descriptions_path),
            "source_url": "https://rh20t.github.io/static/task_description.json",
        },
        "task_policy": {
            "policy_id": task_policy["policy_id"],
            "path": str(task_policy_path) if task_policy_path is not None else None,
            "sha256": sha256_file(task_policy_path) if task_policy_path is not None else None,
            "included_task_count": len(included_tasks),
            "excluded_task_count": len(excluded_tasks),
            "decision_note": task_policy.get("decision_note"),
            "explicitly_included_categories": task_policy.get("explicitly_included_categories", []),
        },
        "config": asdict(config),
        "contract": {
            "robot": "UR5",
            "gripper": "Robotiq 2F-85",
            "joint_order": ACTION_LAYOUT[:6],
            "joint_unit": "radian",
            "action": "future measured absolute joint proxy plus command close fraction",
            "state": "current measured absolute joint plus command close fraction",
            "fixed_fps": config.fps,
            "chunk_length": config.chunk_length,
            "source_nominal_rate_hz": "approximately 7-10; linearly resampled to 15 Hz",
        },
        "counts": {
            "directories": len(directories),
            "auxiliary_directories": auxiliary_count,
            "human_directories": human_count,
            "robot_candidates_scanned": len(candidates),
            "accepted_source_episodes": source_accepted,
            "accepted_segments": len(all_records),
            "rejected_robot_episodes": len(rejected),
            "split_source_episodes": split_sources,
            "split_segments": {split: len(records) for split, records in accepted.items()},
            "split_windows_c32": {
                split: sum(record["window_count_c32"] for record in records) for split, records in accepted.items()
            },
        },
        "rejection_reasons": dict(sorted(reason_counts.items())),
    }
    reports = {
        "schema_gate": {
            **summary,
            "status": "pass" if source_accepted > 0 else "fail",
            "known_limitation": (
                "RH20T has no joint command stream; measured future joints are an explicitly labeled executable proxy."
            ),
        },
        "camera_coverage": {
            "required_serials": EXTERIOR_CAMERAS,
            "accepted_source_episodes": source_accepted,
            "policy": "both external cameras required; no black-frame substitution",
        },
        "gripper_semantics": {
            "status": "pass" if widths.size else "fail",
            "source_field": "transformed/gripper.npy[primary][timestamp].gripper_command[0]",
            "source_unit": "millimeter opening width",
            "observed_quantiles_mm": np.quantile(widths, [0, 0.01, 0.5, 0.99, 1]).tolist() if widths.size else [],
            "canonical_formula": f"close_fraction = 1 - clip(width_mm, 0, {GRIPPER_MAX_WIDTH_MM:g})/{GRIPPER_MAX_WIDTH_MM:g}",
            "gripper_info_note": "gripper_info[0] was constant 84 in the 300-episode probe and is not used.",
        },
        "window_stats_c32": {
            "status": "pass" if all_records else "fail",
            "split_windows": summary["counts"]["split_windows_c32"],
            "movement_steps": dict(movement),
            "movement_fraction": {
                key: float(movement[key] / movement["steps"]) if movement["steps"] else 0.0
                for key in ("hold", "low_motion", "moving")
            },
            "joint_quantiles_rad": {
                ACTION_LAYOUT[index]: np.quantile(joints[:, index], [0, 0.01, 0.5, 0.99, 1]).tolist()
                for index in range(6)
            }
            if joints.size
            else {},
            "window_start_policy": "all legal starts retained; no motion-threshold filtering",
        },
        "train_quantile_7d": {
            "metadata": {
                "dataset": "RH20T cfg4 UR5 joint ext2 fixed15Hz v1",
                "split": "train",
                "action_contract": "future measured absolute joint proxy plus command close fraction",
                "action_dim": 7,
                "action_layout": ACTION_LAYOUT,
                "normalization": "quantile",
                "num_action_rows": int(len(train_actions)),
                "input_fingerprint": summary["input_fingerprint"],
            },
            "global": {
                "mean": train_actions.mean(axis=0).tolist(),
                "std": train_actions.std(axis=0).tolist(),
                "min": train_actions.min(axis=0).tolist(),
                "max": train_actions.max(axis=0).tolist(),
                "q01": np.quantile(train_actions, 0.01, axis=0).tolist(),
                "q99": np.quantile(train_actions, 0.99, axis=0).tolist(),
            }
            if train_actions.size
            else {},
        },
    }
    artifacts = {"inventory": inventory, "rejected": rejected, **accepted}
    task_rows = {
        task_id: {
            "task_id": task_id,
            "instruction_en": row["english"],
            "instruction_zh": row["chinese"],
            "source_episode_count": task_source_counts[task_id],
            "accepted_segment_count": task_segment_counts[task_id],
            "policy_decision": "include" if task_id in included_tasks else "exclude",
            "policy_reason": excluded_tasks.get(task_id, "approved"),
            "review_status": "approved" if task_policy_path is not None else "unfiltered",
            "source": "rh20t.github.io/static/task_description.json",
        }
        for task_id, row in sorted(tasks.items())
        if task_source_counts[task_id]
    }
    return summary, artifacts, {**reports, "task_rows": task_rows}


def _task_catalog_csv(task_rows: dict[str, dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fields = [
        "task_id",
        "instruction_en",
        "instruction_zh",
        "source_episode_count",
        "accepted_segment_count",
        "policy_decision",
        "policy_reason",
        "review_status",
        "source",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(task_rows.values())
    return buffer.getvalue()


def write_outputs(
    output_root: Path,
    summary: dict[str, Any],
    artifacts: dict[str, list[dict[str, Any]]],
    reports: dict[str, Any],
) -> None:
    manifest_root = output_root / "manifests"
    report_root = output_root / "reports"
    stats_root = output_root / "stats"
    windows_root = output_root / "windows_c32"
    atomic_write_json(output_root / "dataset_summary.json", summary)
    atomic_write_jsonl(manifest_root / "source_inventory.jsonl", artifacts["inventory"])
    atomic_write_jsonl(manifest_root / "rejected.jsonl", artifacts["rejected"])
    for split in ("train", "val", "test"):
        records = artifacts[split]
        atomic_write_jsonl(manifest_root / f"{split}.jsonl", records)
        windows = [
            {
                "segment_id": record["segment_id"],
                "split": split,
                "frame_count": record["frame_count"],
                "legal_start_index": [0, record["frame_count"] - CHUNK_LENGTH - 1],
                "window_count": record["window_count_c32"],
            }
            for record in records
        ]
        atomic_write_jsonl(windows_root / f"{split}.jsonl", windows)
    for name in ("schema_gate", "camera_coverage", "gripper_semantics", "window_stats_c32"):
        atomic_write_json(report_root / f"{name}.json", reports[name])
    atomic_write_json(stats_root / "train_quantile_7d.json", reports["train_quantile_7d"])
    atomic_write_text(output_root / "task_catalog.csv", _task_catalog_csv(reports["task_rows"]))
    tracked = [
        output_root / "dataset_summary.json",
        *(manifest_root / f"{split}.jsonl" for split in ("train", "val", "test")),
        stats_root / "train_quantile_7d.json",
        output_root / "task_catalog.csv",
    ]
    atomic_write_json(
        output_root / "checksums.json",
        {
            "input_fingerprint": summary["input_fingerprint"],
            "sha256": {str(path.relative_to(output_root)): sha256_file(path) for path in tracked},
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--task-descriptions", type=Path, default=DEFAULT_TASK_DESCRIPTIONS)
    parser.add_argument("--task-policy", type=Path, default=DEFAULT_TASK_POLICY)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-camera-error-ms", type=float, default=100.0)
    parser.add_argument("--max-interpolation-gap-ms", type=float, default=250.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BuildConfig(
        max_camera_error_ms=args.max_camera_error_ms,
        max_interpolation_gap_ms=args.max_interpolation_gap_ms,
    )
    summary, artifacts, reports = build_manifest(
        source_root=args.source_root,
        task_descriptions_path=args.task_descriptions,
        task_policy_path=args.task_policy,
        config=config,
        limit=args.limit,
    )
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if (
        args.source_root.resolve() == args.output_root.resolve()
        or args.source_root.resolve() in args.output_root.resolve().parents
    ):
        raise ValueError("Output root must not be inside the read-only RH20T source root")
    write_outputs(args.output_root, summary, artifacts, reports)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
