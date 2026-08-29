# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert frozen RH20T cfg4 UR5-joint manifest segments to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import av
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
import numpy as np
from lerobot.datasets.compute_stats import compute_episode_stats as lerobot_compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from tools.rh20t.rh20t_io import (
    ACTION_LAYOUT,
    DEFAULT_DERIVED_ROOT,
    EXTERIOR_CAMERAS,
    FPS,
    PRIMARY_SERIAL,
    atomic_write_json,
    close_fraction,
    interpolate_rows,
    load_color_timestamps,
    load_gripper_width_stream,
    load_joint_stream,
    nearest_indices,
)

DEFAULT_MANIFEST = DEFAULT_DERIVED_ROOT / "manifests" / "train.jsonl"
DEFAULT_OUTPUT = Path("/mnt/pfs/swy/dataset/RH20T/lerobot/rh20t_cfg4_ur5_joint_ext2_15hz_v1.staging")
OnError = Literal["abort", "skip"]


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    manifest: Path = DEFAULT_MANIFEST
    output: Path = DEFAULT_OUTPUT
    repo_id: str = "local/rh20t_cfg4_ur5_joint_ext2_15hz_v1"
    offset: int = 0
    limit: int | None = None
    resume: bool = False
    large_run_ack: bool = False
    on_error: OnError = "abort"
    max_errors: int = 10
    min_free_gb: float = 200.0
    image_writer_threads: int = 4
    parallel_encoding: bool = True
    video_codec: str = "h264"
    progress_every: int = 10


class VideoIndexReader:
    """Read non-decreasing source frame indices with constant memory."""

    def __init__(self, path: Path):
        self.path = path
        self._container = av.open(str(path))
        self._iterator = self._container.decode(video=0)
        self._index = -1
        self._image: np.ndarray | None = None

    def get(self, index: int) -> np.ndarray:
        if index < self._index:
            raise ValueError(f"Video indices must be non-decreasing: requested {index} after {self._index}")
        while self._index < index:
            try:
                frame = next(self._iterator)
            except StopIteration as error:
                raise ValueError(f"Video {self.path} ended before frame {index}") from error
            self._index += 1
            self._image = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
        assert self._image is not None
        return self._image

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> VideoIndexReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    segment_ids = [record.get("segment_id") for record in records]
    if len(segment_ids) != len(set(segment_ids)) or any(not value for value in segment_ids):
        raise ValueError(f"Manifest has missing or duplicate segment_id: {path}")
    return records


def _f32(size: int, names: Any = None) -> dict[str, Any]:
    return {"dtype": "float32", "shape": (size,), "names": names}


def _features() -> dict[str, dict[str, Any]]:
    features = {
        "action": _f32(7, {"motors": ACTION_LAYOUT}),
        "observation.state.joint": _f32(6, {"motors": ACTION_LAYOUT[:6]}),
        "observation.state.gripper": _f32(1, [ACTION_LAYOUT[-1]]),
        # Epoch milliseconds exceed float32 integer precision. LeRobot's torch
        # formatter downcasts floating features, so keep the timestamp integer.
        "source.timestamp_ms": {"dtype": "int64", "shape": (1,), "names": ["timestamp_ms"]},
        "source.camera_alignment_error_ms": _f32(2, list(EXTERIOR_CAMERAS)),
    }
    for key in ("exterior_1", "exterior_2"):
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (360, 640, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _compute_episode_stats_without_images(episode_data: dict, features: dict) -> dict:
    numeric_features = {
        key: feature for key, feature in features.items() if feature.get("dtype") not in ("image", "video", "string")
    }
    return lerobot_compute_episode_stats({key: episode_data[key] for key in numeric_features}, numeric_features)


def _resample_segment(record: dict[str, Any]) -> dict[str, Any]:
    source = Path(record["source_path"])
    expected = int(record["frame_count"])
    target = float(record["target_start_ms"]) + np.arange(expected) * float(record["target_step_ms"])
    transformed = source / "transformed"
    joint_ts, joints = load_joint_stream(transformed / "joint.npy", PRIMARY_SERIAL)
    gripper_ts, width_mm = load_gripper_width_stream(transformed / "gripper.npy", PRIMARY_SERIAL)
    joint = interpolate_rows(joint_ts, np.unwrap(joints, axis=0), target).astype(np.float32)
    gripper = close_fraction(interpolate_rows(gripper_ts, width_mm, target)).astype(np.float32)
    if not np.isfinite(joint).all() or not np.isfinite(gripper).all():
        raise ValueError(f"Non-finite resampled state in {record['segment_id']}")
    if np.max(np.abs(joint)) > 2 * np.pi + 0.25:
        raise ValueError(f"Joint magnitude exceeds UR5 gate in {record['segment_id']}")
    if np.any((gripper < 0) | (gripper > 1)):
        raise ValueError(f"Gripper close_fraction leaves [0,1] in {record['segment_id']}")

    camera_indices: dict[str, np.ndarray] = {}
    camera_errors: dict[str, np.ndarray] = {}
    for serial in EXTERIOR_CAMERAS:
        timestamps = load_color_timestamps(source / f"cam_{serial}" / "timestamps.npy")
        indices, errors = nearest_indices(timestamps, target)
        camera_indices[serial] = indices
        camera_errors[serial] = errors.astype(np.float32)
        if errors.max(initial=0.0) > 100.0 + 1e-4:
            raise ValueError(f"Camera {serial} alignment exceeds 100 ms in {record['segment_id']}")
    return {
        "target_timestamps": target,
        "joint": joint,
        "gripper": gripper,
        "camera_indices": camera_indices,
        "camera_errors": camera_errors,
    }


def _add_segment(dataset: LeRobotDataset, record: dict[str, Any], *, parallel_encoding: bool) -> int:
    arrays = _resample_segment(record)
    source = Path(record["source_path"])
    readers = {serial: VideoIndexReader(source / f"cam_{serial}" / "color.mp4") for serial in EXTERIOR_CAMERAS}
    try:
        for index in range(int(record["frame_count"])):
            joint = arrays["joint"][index]
            gripper = np.asarray([arrays["gripper"][index]], dtype=np.float32)
            action = np.concatenate((joint, gripper)).astype(np.float32)
            frame = {
                "task": str(record["task_instruction_en"]),
                "action": action,
                "observation.state.joint": joint,
                "observation.state.gripper": gripper,
                "source.timestamp_ms": np.asarray([round(float(arrays["target_timestamps"][index]))], dtype=np.int64),
                "source.camera_alignment_error_ms": np.asarray(
                    [arrays["camera_errors"][serial][index] for serial in EXTERIOR_CAMERAS], dtype=np.float32
                ),
                "observation.images.exterior_1": readers[EXTERIOR_CAMERAS[0]].get(
                    int(arrays["camera_indices"][EXTERIOR_CAMERAS[0]][index])
                ),
                "observation.images.exterior_2": readers[EXTERIOR_CAMERAS[1]].get(
                    int(arrays["camera_indices"][EXTERIOR_CAMERAS[1]][index])
                ),
            }
            dataset.add_frame(frame)
    finally:
        for reader in readers.values():
            reader.close()
    dataset.save_episode(parallel_encoding=parallel_encoding)
    return int(record["frame_count"])


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _saved_segments(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(record["segment_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (record := json.loads(line)).get("status") == "saved"
    }


def _free_gb(path: Path) -> float:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def _make_dataset(config: ConversionConfig) -> LeRobotDataset:
    if config.resume:
        dataset = LeRobotDataset(
            repo_id=config.repo_id,
            root=config.output,
            revision="local",
            vcodec=config.video_codec,
        )
        if float(dataset.meta.fps) != FPS:
            raise ValueError(f"Resume dataset fps={dataset.meta.fps}, expected {FPS}")
        return dataset
    return LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=FPS,
        features=_features(),
        root=config.output,
        robot_type="ur5",
        use_videos=True,
        image_writer_threads=config.image_writer_threads,
        batch_encoding_size=1,
        vcodec=config.video_codec,
    )


def convert(config: ConversionConfig) -> dict[str, Any]:
    if config.output.exists() and not config.resume:
        raise FileExistsError(f"Output exists: {config.output}; use a fresh staging path or --resume")
    if config.resume and not config.output.is_dir():
        raise FileNotFoundError(f"Resume output does not exist: {config.output}")
    if config.limit is None and not config.large_run_ack:
        raise ValueError("Uncapped conversion requires --large-run-ack")
    if _free_gb(config.output) < config.min_free_gb:
        raise RuntimeError(f"Free space below min_free_gb={config.min_free_gb} at {config.output}")

    if config.offset < 0:
        raise ValueError("offset must be non-negative")
    all_records = _read_jsonl(config.manifest)
    if (config.limit is None or config.large_run_ack) and any(
        record.get("task_policy_decision") != "include" or not record.get("task_policy_id") for record in all_records
    ):
        raise ValueError("Uncapped conversion requires a reviewed task-policy manifest")
    records = list(enumerate(all_records))[config.offset :]
    if config.limit is not None:
        records = records[: config.limit]
    run_root = config.output / "conversion"
    ledger = run_root / "ledger.jsonl"
    errors = run_root / "errors.jsonl"
    saved = _saved_segments(ledger) if config.resume else set()
    if config.resume:
        try:
            metadata = LeRobotDatasetMetadata(
                repo_id=config.repo_id,
                root=config.output,
                revision="local",
            )
        except FileNotFoundError as error:
            raise ValueError("Resume output has no completed episode metadata; use a fresh staging path") from error
        if len(saved) != int(metadata.total_episodes):
            raise ValueError("Resume ledger count does not match LeRobot episode count")

    lerobot_dataset_module.compute_episode_stats = _compute_episode_stats_without_images
    dataset = _make_dataset(config)
    started = time.time()
    saved_now = 0
    frames = 0
    error_count = 0
    status = "running"
    try:
        for manifest_index, record in records:
            segment_id = str(record["segment_id"])
            if segment_id in saved:
                continue
            try:
                frame_count = _add_segment(dataset, record, parallel_encoding=config.parallel_encoding)
            except Exception as error:
                if dataset.episode_buffer and dataset.episode_buffer.get("size", 0):
                    dataset.clear_episode_buffer(delete_images=True)
                error_count += 1
                _append_jsonl(
                    errors,
                    {
                        "segment_id": segment_id,
                        "manifest_index": manifest_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                if config.on_error == "abort" or error_count >= config.max_errors:
                    raise
                continue
            saved_now += 1
            frames += frame_count
            _append_jsonl(
                ledger,
                {
                    "status": "saved",
                    "segment_id": segment_id,
                    "manifest_index": manifest_index,
                    "lerobot_episode_index": int(dataset.meta.total_episodes) - 1,
                    "frames": frame_count,
                },
            )
            if saved_now == 1 or saved_now % max(1, config.progress_every) == 0:
                print(
                    f"RH20T conversion progress: saved_now={saved_now}/{len(records)} frames={frames} "
                    f"segment={segment_id}",
                    flush=True,
                )
            if _free_gb(config.output) < config.min_free_gb:
                raise RuntimeError(f"Free space fell below min_free_gb={config.min_free_gb}")
        dataset.finalize()
        status = "complete"
    finally:
        report = {
            "status": status,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "manifest_records_selected": len(records),
            "previously_saved": len(saved),
            "saved_now": saved_now,
            "frames_saved_now": frames,
            "errors": error_count,
            "elapsed_s": round(time.time() - started, 3),
            "output": str(config.output),
        }
        atomic_write_json(run_root / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="local/rh20t_cfg4_ur5_joint_ext2_15hz_v1")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--large-run-ack", action="store_true")
    parser.add_argument("--on-error", choices=("abort", "skip"), default="abort")
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--min-free-gb", type=float, default=200.0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--parallel-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-codec", choices=("h264", "hevc", "libsvtav1"), default="h264")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ConversionConfig(
        manifest=args.manifest,
        output=args.output,
        repo_id=args.repo_id,
        offset=args.offset,
        limit=args.limit,
        resume=args.resume,
        large_run_ack=args.large_run_ack,
        on_error=args.on_error,
        max_errors=args.max_errors,
        min_free_gb=args.min_free_gb,
        image_writer_threads=args.image_writer_threads,
        parallel_encoding=args.parallel_encoding,
        video_codec=args.video_codec,
        progress_every=args.progress_every,
    )
    print(json.dumps(convert(config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
