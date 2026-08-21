# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""F2 trajectory and video quality gates for the DROID Fx4 pipeline."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from tools.droid.fx4_io import json_fingerprint, load_jsonl, sha256_file, write_checksums, write_json, write_jsonl

CAMERA_KEYS = (
    "observation.image.wrist_image_left",
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
)
POSE_FEATURE = "observation.state.cartesian_position"
GRIPPER_FEATURE = "action.gripper_position"
SMOOTHING_RULE = "se3_local_geodesic_binomial_w5_endpoints_fixed_v1"


@dataclass(frozen=True)
class F2Config:
    fps: float = 15.0
    timestamp_tolerance_s: float = 2e-4
    chunk_length: int = 32
    smoothing_window: int = 5
    max_step_translation_m: float = 0.10
    max_step_rotation_deg: float = 45.0
    max_smoothing_translation_m: float = 0.02
    max_smoothing_rotation_deg: float = 10.0
    black_mean_max: float = 5.0
    black_std_max: float = 2.0
    frozen_mad_max: float = 0.20
    frozen_ratio_max: float = 0.90


@dataclass(frozen=True)
class Trajectory:
    timestamp: np.ndarray
    frame_index: np.ndarray
    raw_xyz: np.ndarray
    raw_rotation: Rotation
    smooth_xyz: np.ndarray
    smooth_rotation: Rotation
    gripper_close_fraction: np.ndarray


def smooth_se3(xyz: np.ndarray, rotation: Rotation, window: int = 5) -> tuple[np.ndarray, Rotation]:
    """Smooth translation and rotation locally on SE(3), preserving endpoints."""
    if window < 3 or window % 2 == 0:
        raise ValueError("smoothing window must be odd and at least 3")
    if len(xyz) != len(rotation):
        raise ValueError("translation and rotation lengths differ")
    if len(xyz) <= 2:
        return xyz.copy(), Rotation.from_quat(rotation.as_quat())

    radius = window // 2
    smooth_xyz = xyz.astype(np.float64, copy=True)
    smooth_matrices = rotation.as_matrix().copy()
    full_weights = np.arange(1, radius + 2, dtype=np.float64)
    full_weights = np.concatenate((full_weights, full_weights[-2::-1]))
    for index in range(1, len(xyz) - 1):
        start = max(0, index - radius)
        stop = min(len(xyz), index + radius + 1)
        weight_start = radius - (index - start)
        weights = full_weights[weight_start : weight_start + stop - start]
        weights /= weights.sum()
        smooth_xyz[index] = np.sum(xyz[start:stop] * weights[:, None], axis=0)

        center = rotation[index]
        relative = center.inv() * rotation[start:stop]
        correction = np.sum(relative.as_rotvec() * weights[:, None], axis=0)
        smooth_matrices[index] = (center * Rotation.from_rotvec(correction)).as_matrix()
    return smooth_xyz, Rotation.from_matrix(smooth_matrices)


def rotation_step_degrees(rotation: Rotation) -> np.ndarray:
    if len(rotation) < 2:
        return np.empty(0, dtype=np.float64)
    return np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())


def build_trajectory(
    *, timestamp: np.ndarray, frame_index: np.ndarray, state: np.ndarray, gripper: np.ndarray, config: F2Config
) -> Trajectory:
    timestamp = np.asarray(timestamp, dtype=np.float64)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    state = np.asarray(state, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    lengths = {len(timestamp), len(frame_index), len(state), len(gripper)}
    if len(lengths) != 1 or state.ndim != 2 or state.shape[1] != 6:
        raise ValueError("invalid DROID trajectory shapes")
    raw_rotation = Rotation.from_euler("xyz", state[:, 3:6])
    smooth_xyz, smooth_rotation = smooth_se3(state[:, :3], raw_rotation, config.smoothing_window)
    return Trajectory(
        timestamp=timestamp,
        frame_index=frame_index,
        raw_xyz=state[:, :3],
        raw_rotation=raw_rotation,
        smooth_xyz=smooth_xyz,
        smooth_rotation=smooth_rotation,
        gripper_close_fraction=gripper,
    )


def trajectory_reason_codes(trajectory: Trajectory, config: F2Config) -> list[str]:
    reasons: list[str] = []
    arrays = (
        trajectory.timestamp,
        trajectory.raw_xyz,
        trajectory.raw_rotation.as_quat(),
        trajectory.gripper_close_fraction,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        reasons.append("non_finite")
    if len(trajectory.timestamp) > 1:
        delta = np.diff(trajectory.timestamp)
        if np.any(delta <= 0):
            reasons.append("timestamp_not_monotonic")
        if np.max(np.abs(delta - 1.0 / config.fps)) > config.timestamp_tolerance_s:
            reasons.append("timestamp_not_15hz")
    if not np.array_equal(trajectory.frame_index, np.arange(len(trajectory.frame_index))):
        reasons.append("frame_index_not_contiguous")
    return reasons


def range_frame_bounds(range_record: dict[str, Any], length: int, chunk_length: int) -> tuple[int, int]:
    start = int(range_record["clipped_start"])
    stop = int(range_record["clipped_end"]) + chunk_length
    if int(range_record["window_count"]) <= 0 or start < 0 or stop > length or stop - start < chunk_length + 1:
        raise ValueError(f"Invalid trainable range bounds [{start}, {stop}) for length={length}")
    return start, stop


def motion_metrics(trajectory: Trajectory, start: int, stop: int, fps: float) -> dict[str, float]:
    raw_xyz = trajectory.raw_xyz[start:stop]
    raw_rotation = trajectory.raw_rotation[start:stop]
    smooth_xyz = trajectory.smooth_xyz[start:stop]
    smooth_rotation = trajectory.smooth_rotation[start:stop]
    translation_step = np.linalg.norm(np.diff(raw_xyz, axis=0), axis=1)
    rotation_step = rotation_step_degrees(raw_rotation)
    smoothing_translation = np.linalg.norm(raw_xyz - smooth_xyz, axis=1)
    smoothing_rotation = np.degrees((raw_rotation.inv() * smooth_rotation).magnitude())
    gripper = trajectory.gripper_close_fraction[start:stop]
    return {
        "max_step_translation_m": float(translation_step.max(initial=0.0)),
        "max_step_rotation_deg": float(rotation_step.max(initial=0.0)),
        "max_speed_m_s": float(translation_step.max(initial=0.0) * fps),
        "path_length_m": float(translation_step.sum()),
        "max_smoothing_translation_m": float(smoothing_translation.max(initial=0.0)),
        "max_smoothing_rotation_deg": float(smoothing_rotation.max(initial=0.0)),
        "gripper_min": float(gripper.min(initial=math.inf)),
        "gripper_max": float(gripper.max(initial=-math.inf)),
    }


def motion_gate_reasons(metrics: dict[str, float], config: F2Config) -> list[str]:
    checks = (
        ("step_translation_above_absolute_limit", "max_step_translation_m", config.max_step_translation_m),
        ("step_rotation_above_absolute_limit", "max_step_rotation_deg", config.max_step_rotation_deg),
        (
            "smoothing_translation_above_limit",
            "max_smoothing_translation_m",
            config.max_smoothing_translation_m,
        ),
        ("smoothing_rotation_above_limit", "max_smoothing_rotation_deg", config.max_smoothing_rotation_deg),
    )
    return [reason for reason, key, limit in checks if metrics[key] > limit]


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "p99": 0.0, "p999": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "max": float(values.max()),
    }


def _metadata_columns() -> list[str]:
    columns = ["episode_index", "episode_id", "length", "data/chunk_index", "data/file_index"]
    for camera in CAMERA_KEYS:
        columns.extend(
            [
                f"videos/{camera}/chunk_index",
                f"videos/{camera}/file_index",
                f"videos/{camera}/from_timestamp",
                f"videos/{camera}/to_timestamp",
            ]
        )
    return columns


def load_episode_metadata(success_root: Path) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for path in sorted((success_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path, columns=_metadata_columns()).to_pylist():
            index = int(row["episode_index"])
            if index in metadata:
                raise ValueError(f"Duplicate episode metadata: {index}")
            metadata[index] = row
    return metadata


def _accepted_f1_records(f1_path: Path) -> dict[int, dict[str, Any]]:
    return {int(record["episode_index"]): record for record in load_jsonl(f1_path) if record["f1_status"] == "accepted"}


def load_accepted_trajectories(
    success_root: Path, accepted: dict[int, dict[str, Any]], metadata: dict[int, dict[str, Any]], config: F2Config
) -> dict[int, Trajectory]:
    by_file: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for episode_index in accepted:
        row = metadata[episode_index]
        by_file[(int(row["data/chunk_index"]), int(row["data/file_index"]))].append(episode_index)

    trajectories: dict[int, Trajectory] = {}
    columns = ["episode_index", "timestamp", "frame_index", POSE_FEATURE, GRIPPER_FEATURE]
    for file_number, ((chunk_index, file_index), episode_indices) in enumerate(sorted(by_file.items()), start=1):
        path = success_root / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        table = pq.read_table(path, columns=columns)
        table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(episode_indices, type=pa.int64())))
        rows_by_episode: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in table.to_pylist():
            rows_by_episode[int(row["episode_index"])].append(row)
        for episode_index in episode_indices:
            rows = rows_by_episode[episode_index]
            if len(rows) != int(metadata[episode_index]["length"]):
                raise ValueError(f"Episode {episode_index} data length mismatch")
            trajectories[episode_index] = build_trajectory(
                timestamp=np.asarray([row["timestamp"] for row in rows]),
                frame_index=np.asarray([row["frame_index"] for row in rows]),
                state=np.asarray([row[POSE_FEATURE] for row in rows]),
                gripper=np.asarray([row[GRIPPER_FEATURE] for row in rows]),
                config=config,
            )
        print(f"F2 motion load: {file_number}/{len(by_file)} data shards", flush=True)
    return trajectories


def build_f2_motion(
    *,
    success_root: Path,
    f1_dir: Path,
    output_dir: Path,
    config: F2Config,
    limit_episodes: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    f1_path = f1_dir / "f1_episodes.jsonl"
    f1_summary_path = f1_dir / "f1_summary.json"
    accepted = _accepted_f1_records(f1_path)
    if limit_episodes is not None:
        accepted = dict(list(sorted(accepted.items()))[:limit_episodes])
    metadata = load_episode_metadata(success_root)
    trajectories = load_accepted_trajectories(success_root, accepted, metadata, config)

    distributions: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for episode_index, record in accepted.items():
        trajectory = trajectories[episode_index]
        for range_record in record["ranges"]:
            if int(range_record["window_count"]) <= 0:
                continue
            start, stop = range_frame_bounds(range_record, len(trajectory.timestamp), config.chunk_length)
            raw_xyz = trajectory.raw_xyz[start:stop]
            raw_rotation = trajectory.raw_rotation[start:stop]
            distributions["step_translation_m"].append(np.linalg.norm(np.diff(raw_xyz, axis=0), axis=1))
            distributions["step_rotation_deg"].append(rotation_step_degrees(raw_rotation))
            distributions["smoothing_translation_m"].append(
                np.linalg.norm(raw_xyz - trajectory.smooth_xyz[start:stop], axis=1)
            )
            distributions["smoothing_rotation_deg"].append(
                np.degrees((raw_rotation.inv() * trajectory.smooth_rotation[start:stop]).magnitude())
            )

    thresholds = {
        "version": "fx4_f2_thresholds_v1",
        "pose_smoothing_rule": SMOOTHING_RULE,
        "config": config.__dict__,
        "observed": {
            name: distribution_summary(np.concatenate(values) if values else np.empty(0))
            for name, values in sorted(distributions.items())
        },
        "policy": "P99/P99.9 are audit thresholds; only explicit absolute limits reject in F2 v1.",
    }

    records: list[dict[str, Any]] = []
    status_counts: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    kept_ranges = 0
    kept_windows = 0
    for episode_index, f1_record in sorted(accepted.items()):
        trajectory = trajectories[episode_index]
        episode_reasons = trajectory_reason_codes(trajectory, config)
        ranges: list[dict[str, Any]] = []
        for range_record in f1_record["ranges"]:
            if int(range_record["window_count"]) <= 0:
                continue
            start, stop = range_frame_bounds(range_record, len(trajectory.timestamp), config.chunk_length)
            metrics = motion_metrics(trajectory, start, stop, config.fps)
            reasons = [*episode_reasons, *motion_gate_reasons(metrics, config)]
            status = "accepted" if not reasons else "rejected"
            if status == "accepted":
                kept_ranges += 1
                kept_windows += int(range_record["window_count"])
            else:
                reason_counts.update(reasons)
            ranges.append(
                {
                    **range_record,
                    "pose_frame_start": start,
                    "pose_frame_stop": stop,
                    "motion_metrics": metrics,
                    "f2_motion_status": status,
                    "f2_motion_reason_codes": reasons,
                }
            )
        episode_status = "accepted" if any(item["f2_motion_status"] == "accepted" for item in ranges) else "rejected"
        status_counts[episode_status] += 1
        meta = metadata[episode_index]
        records.append(
            {
                "episode_index": episode_index,
                "episode_id": f1_record["episode_id"],
                "length": f1_record["length"],
                "task_family": f1_record["task_family"],
                "tasks_raw": f1_record["tasks_raw"],
                "tasks_annotations": f1_record["tasks_annotations"],
                "data_chunk_index": int(meta["data/chunk_index"]),
                "data_file_index": int(meta["data/file_index"]),
                "video_metadata": {
                    camera: {
                        "chunk_index": int(meta[f"videos/{camera}/chunk_index"]),
                        "file_index": int(meta[f"videos/{camera}/file_index"]),
                        "from_timestamp": float(meta[f"videos/{camera}/from_timestamp"]),
                        "to_timestamp": float(meta[f"videos/{camera}/to_timestamp"]),
                    }
                    for camera in CAMERA_KEYS
                },
                "pose_smoothing_rule": SMOOTHING_RULE,
                "force_event_source": "unavailable",
                "ranges": ranges,
                "f2_motion_status": episode_status,
            }
        )

    inputs = {
        "f1_summary": {"path": str(f1_summary_path), "sha256": sha256_file(f1_summary_path)},
        "f1_episodes": {"path": str(f1_path), "sha256": sha256_file(f1_path)},
    }
    summary = {
        "stage": "f2_motion",
        "limit_episodes": limit_episodes,
        "counts": {
            "processed_episodes": len(records),
            "status": dict(sorted(status_counts.items())),
            "kept_ranges": kept_ranges,
            "kept_windows": kept_windows,
            "rejection_reasons": dict(sorted(reason_counts.items())),
        },
        "inputs": inputs,
        "thresholds_sha256": json_fingerprint(thresholds),
    }
    summary["input_fingerprint"] = json_fingerprint(
        {
            "inputs": inputs,
            "thresholds": thresholds,
            "output_dir": str(output_dir.resolve()),
            "limit_episodes": limit_episodes,
        }
    )
    return summary, records, thresholds


def write_f2_motion(
    output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]], thresholds: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "f2_motion_summary.json"
    records_path = output_dir / "f2_motion_episodes.jsonl"
    thresholds_path = output_dir / "f2_thresholds.json"
    write_json(summary_path, summary)
    write_jsonl(records_path, records)
    write_json(thresholds_path, thresholds)
    write_checksums(output_dir / "F2_MOTION_SHA256SUMS", (summary_path, records_path, thresholds_path))


def _video_path(success_root: Path, camera: str, metadata: dict[str, Any]) -> Path:
    return success_root / (f"videos/{camera}/chunk-{metadata['chunk_index']:03d}/file-{metadata['file_index']:03d}.mp4")


def decode_video_samples(
    path: Path, timestamps: list[float], config: F2Config, downsample_stride: int = 10
) -> list[dict[str, Any]]:
    """Decode nearest frames at absolute container timestamps."""
    if not path.is_file():
        return [{"status": "missing"} for _ in timestamps]
    results: list[dict[str, Any]] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for target in timestamps:
                container.seek(max(0, int(target / float(stream.time_base))), stream=stream, backward=True)
                selected = None
                for frame in container.decode(stream):
                    frame_time = float(frame.time or 0.0)
                    selected = frame
                    if frame_time >= target - 0.5 / config.fps:
                        break
                if selected is None:
                    results.append({"status": "decode_failure"})
                    continue
                image = selected.to_ndarray(format="rgb24")[::downsample_stride, ::downsample_stride]
                mean = float(image.mean())
                std = float(image.std())
                results.append(
                    {
                        "status": "black" if mean <= config.black_mean_max and std <= config.black_std_max else "ok",
                        "target_timestamp": target,
                        "decoded_timestamp": float(selected.time or 0.0),
                        "timestamp_error_s": abs(float(selected.time or 0.0) - target),
                        "mean": mean,
                        "std": std,
                        "thumbnail": image,
                    }
                )
    except (av.FFmpegError, OSError, ValueError):
        return [{"status": "container_failure"} for _ in timestamps]
    return results


def range_sample_frames(start: int, stop: int) -> list[int]:
    if stop <= start:
        raise ValueError("empty video range")
    return sorted(set(int(round(value)) for value in np.linspace(start, stop - 1, 5)))


def sample_freeze_ratio(samples: list[dict[str, Any]], config: F2Config) -> float:
    images = [sample.get("thumbnail") for sample in samples if sample.get("status") == "ok"]
    if len(images) < 2:
        return 0.0
    frozen = [
        float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))) <= config.frozen_mad_max
        for left, right in zip(images, images[1:])
    ]
    return float(np.mean(frozen))


def build_f2_video(
    *, success_root: Path, f2_dir: Path, config: F2Config, limit_episodes: int | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    motion_path = f2_dir / "f2_motion_episodes.jsonl"
    motion_summary_path = f2_dir / "f2_motion_summary.json"
    records = [record for record in load_jsonl(motion_path) if record["f2_motion_status"] == "accepted"]
    if limit_episodes is not None:
        records = records[:limit_episodes]

    output_records: list[dict[str, Any]] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    status_counts: collections.Counter[str] = collections.Counter()
    camera_sample_counts: collections.Counter[str] = collections.Counter()
    camera_failure_counts: collections.Counter[str] = collections.Counter()
    kept_ranges = 0
    kept_windows = 0
    for episode_number, record in enumerate(records, start=1):
        ranges: list[dict[str, Any]] = []
        for range_record in record["ranges"]:
            if range_record["f2_motion_status"] != "accepted":
                ranges.append(range_record)
                continue
            frames = range_sample_frames(range_record["pose_frame_start"], range_record["pose_frame_stop"])
            reasons: list[str] = []
            camera_results: dict[str, Any] = {}
            for camera in CAMERA_KEYS:
                video_meta = record["video_metadata"][camera]
                path = _video_path(success_root, camera, video_meta)
                timestamps = [video_meta["from_timestamp"] + frame / config.fps for frame in frames]
                samples = decode_video_samples(path, timestamps, config)
                camera_sample_counts[camera] += len(samples)
                statuses = [sample["status"] for sample in samples]
                for status in statuses:
                    if status != "ok":
                        camera_failure_counts[f"{camera}:{status}"] += 1
                if any(status in {"missing", "container_failure", "decode_failure"} for status in statuses):
                    reasons.append(f"video_decode_failure:{camera}")
                if "black" in statuses:
                    reasons.append(f"video_black_frame:{camera}")
                freeze_ratio = sample_freeze_ratio(samples, config)
                if freeze_ratio > config.frozen_ratio_max:
                    reasons.append(f"video_frozen:{camera}")
                camera_results[camera] = {
                    "path": str(path),
                    "sample_frames": frames,
                    "statuses": statuses,
                    "max_timestamp_error_s": max(
                        (sample.get("timestamp_error_s", 0.0) for sample in samples), default=0.0
                    ),
                    "sample_freeze_ratio": freeze_ratio,
                }
            status = "accepted" if not reasons else "rejected"
            if status == "accepted":
                kept_ranges += 1
                kept_windows += int(range_record["window_count"])
            else:
                reason_counts.update(reasons)
            ranges.append(
                {
                    **range_record,
                    "video_quality": camera_results,
                    "f2_status": status,
                    "f2_reason_codes": reasons,
                }
            )
        episode_status = "accepted" if any(item.get("f2_status") == "accepted" for item in ranges) else "rejected"
        status_counts[episode_status] += 1
        output_records.append({**record, "ranges": ranges, "f2_status": episode_status})
        if episode_number % 100 == 0:
            print(f"F2 video progress: {episode_number}/{len(records)} episodes", flush=True)

    total_video_samples = sum(camera_sample_counts.values())
    total_failures = sum(camera_failure_counts.values())
    decode_failure_rate = total_failures / total_video_samples if total_video_samples else 1.0
    inputs = {
        "motion_summary": {"path": str(motion_summary_path), "sha256": sha256_file(motion_summary_path)},
        "motion_episodes": {"path": str(motion_path), "sha256": sha256_file(motion_path)},
    }
    summary = {
        "stage": "f2",
        "limit_episodes": limit_episodes,
        "counts": {
            "processed_episodes": len(output_records),
            "status": dict(sorted(status_counts.items())),
            "kept_ranges": kept_ranges,
            "kept_windows": kept_windows,
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "camera_samples": dict(sorted(camera_sample_counts.items())),
            "camera_failures": dict(sorted(camera_failure_counts.items())),
        },
        "gates": {
            "video_decode_failure_rate": decode_failure_rate,
            "video_decode_failure_lt_0_001": decode_failure_rate < 0.001,
        },
        "inputs": inputs,
    }
    summary["input_fingerprint"] = json_fingerprint(
        {"inputs": inputs, "config": config.__dict__, "limit": limit_episodes}
    )
    return summary, output_records


def write_f2_video(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "f2_summary.json"
    records_path = output_dir / "f2_episodes.jsonl"
    write_json(summary_path, summary)
    write_jsonl(records_path, records)
    write_checksums(output_dir / "F2_SHA256SUMS", (summary_path, records_path, output_dir / "f2_thresholds.json"))
