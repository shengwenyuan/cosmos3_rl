#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Materialize a joint-action RoboLabSim-147 LeRobot sibling dataset.

The existing ``robolabsim-147`` root already stores the native 7D UR5 joint
command in ``action``. The EEF training path derives a 10D EEF delta target in
the Cosmos dataset adapter. This script keeps the native joint target, removes
auxiliary action-space pose fields by default, and validates the result against
the raw keypoint-IK HDF5 files when available.
"""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_SRC = Path("/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robolabsim-147")
DEFAULT_OUT = Path("/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robolabsim-joints-147")
DEFAULT_RAW_ROOT = Path("/mlp_vepfs/share/swy/cosmos3-framework/keypoint_ik_runs/robolabsim-147/raw")

AUX_ACTION_FIELDS = ("action.joint_position", "action.tool0_pose")
REQUIRED_WIDTHS = {
    "action": 7,
    "observation.state": 12,
    "observation.velocity": 12,
    "observation.ee_position": 3,
    "observation.ee_orientation": 4,
}
SCALAR_FEATURES = ("episode_index", "frame_index", "index", "task_index", "timestamp")
VIDEO_PREFIX = "observation.images."
RAW_TASK_RE = re.compile(
    r"^\d+_(?P<object>.+)_to_(?P<target>.+)_(original|random_background_lighting|random_table_material)$"
)


@dataclass(frozen=True)
class RawCandidate:
    path: Path
    task: str | None
    action: np.ndarray
    state: np.ndarray
    velocity: np.ndarray
    ee_position: np.ndarray
    ee_orientation: np.ndarray
    raw_action_joint_max_abs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Existing RoboLabSim LeRobot root.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output joint-action LeRobot root.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Raw keypoint-IK root for validation.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize video files in the output root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output root before writing.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Write only the first N episodes for smoke tests.")
    parser.add_argument(
        "--keep-aux-action-fields",
        action="store_true",
        help="Keep action.joint_position and action.tool0_pose in the output parquet/meta.",
    )
    parser.add_argument(
        "--raw-validate",
        choices=("all", "sample", "none"),
        default="all",
        help="Validate output numeric streams against raw HDF5 files.",
    )
    parser.add_argument("--raw-sample-episodes", type=int, default=8, help="Episode count for --raw-validate=sample.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Max absolute tolerance for raw action checks.")
    parser.add_argument(
        "--state-tolerance",
        type=float,
        default=1e-5,
        help="Max absolute tolerance for raw state/EEF checks.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def source_data_path(root: Path, info: dict[str, Any]) -> Path:
    return root / info["data_path"].format(chunk_index=0, file_index=0)


def source_episode_path(root: Path) -> Path:
    return root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"


def dst_data_path(root: Path, info: dict[str, Any]) -> Path:
    return root / info["data_path"].format(chunk_index=0, file_index=0)


def dst_episode_path(root: Path) -> Path:
    return root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"


def vector_width(table: pa.Table, key: str) -> int | None:
    if key not in table.column_names or table.num_rows == 0:
        return None
    first = table.column(key).combine_chunks()[0].as_py()
    if isinstance(first, list):
        return len(first)
    return 1


def assert_feature_widths(table: pa.Table, info: dict[str, Any]) -> None:
    features = info.get("features", {})
    for key, width in REQUIRED_WIDTHS.items():
        if key not in features:
            raise ValueError(f"Missing required feature in info.json: {key}")
        shape = features[key].get("shape")
        if not isinstance(shape, list) or shape[:1] != [width]:
            raise ValueError(f"Feature {key} has shape {shape}, expected [{width}]")
        actual = vector_width(table, key)
        if actual != width:
            raise ValueError(f"Parquet column {key} width {actual}, expected {width}")


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def filter_first_episodes(table: pa.Table, episodes: pa.Table, max_episodes: int | None) -> tuple[pa.Table, pa.Table]:
    if max_episodes is None:
        return table, episodes
    if max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    data_mask = pc.less(table["episode_index"], pa.scalar(max_episodes, pa.int64()))
    ep_mask = pc.less(episodes["episode_index"], pa.scalar(max_episodes, pa.int64()))
    return table.filter(data_mask), episodes.filter(ep_mask)


def drop_aux_action_columns(table: pa.Table, info: dict[str, Any], keep_aux: bool) -> tuple[pa.Table, list[str]]:
    if keep_aux:
        return table, []
    removed: list[str] = []
    for key in AUX_ACTION_FIELDS:
        if key in table.column_names:
            table = table.drop([key])
            removed.append(key)
        info.get("features", {}).pop(key, None)
    return table, removed


def feature_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video" and key.startswith(VIDEO_PREFIX)
    ]


def copy_regular_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def link_or_copy_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "symlink":
        os.symlink(src, dst)
        return "symlink"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise OSError(f"Hardlink failed across filesystems for {src} -> {dst}; use --link-mode symlink or copy") from exc
        raise


def materialize_meta_files(src: Path, out: Path) -> None:
    for rel in (
        Path("meta/tasks.jsonl"),
        Path("meta/tasks.parquet"),
        Path("meta/cameras.json"),
        Path("keypoint_ik_validation_report.json"),
    ):
        source = src / rel
        if source.exists():
            copy_regular_file(source, out / rel)


def materialize_videos(
    src: Path,
    out: Path,
    info: dict[str, Any],
    episodes: pa.Table,
    link_mode: str,
) -> dict[str, Any]:
    video_keys = feature_video_keys(info)
    ep_rows = episodes.to_pylist()
    links: list[dict[str, Any]] = []
    for row in ep_rows:
        for key in video_keys:
            chunk_index = int(row[f"videos/{key}/chunk_index"])
            file_index = int(row[f"videos/{key}/file_index"])
            rel = Path(info["video_path"].format(video_key=key, chunk_index=chunk_index, file_index=file_index))
            source = src / rel
            if not source.is_file():
                raise FileNotFoundError(source)
            method = link_or_copy_file(source, out / rel, link_mode)
            links.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "video_key": key,
                    "path": str(rel),
                    "bytes": source.stat().st_size,
                    "method": method,
                }
            )
    return {
        "video_keys": video_keys,
        "count": len(links),
        "bytes": int(sum(item["bytes"] for item in links)),
        "methods": sorted({item["method"] for item in links}),
        "preview": links[:5],
    }


def array_from_column(table: pa.Table, key: str, dtype: type = np.float64) -> np.ndarray:
    values = table.column(key).combine_chunks().to_pylist()
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def compute_numeric_stats(table: pa.Table, info: dict[str, Any], src_stats: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    total_frames = table.num_rows
    for key, feature in info.get("features", {}).items():
        dtype = feature.get("dtype")
        if dtype == "video":
            video_stats = copy.deepcopy(src_stats.get(key))
            if video_stats is None:
                video_stats = {
                    "mean": [[[0.5]], [[0.5]], [[0.5]]],
                    "std": [[[0.5]], [[0.5]], [[0.5]]],
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[1.0]], [[1.0]], [[1.0]]],
                }
            video_stats["count"] = [total_frames]
            stats[key] = video_stats
            continue
        if key not in table.column_names or dtype == "bool":
            continue
        arr = array_from_column(table, key)
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in {key}")
        stats[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
    return stats


def total_file_size(root: Path, rel_glob: str) -> int:
    return int(sum(path.stat().st_size for path in root.glob(rel_glob) if path.is_file()))


def update_info_counts(info: dict[str, Any], out: Path, table: pa.Table, episodes: pa.Table, video_report: dict[str, Any]) -> None:
    info["total_episodes"] = int(episodes.num_rows)
    info["total_frames"] = int(table.num_rows)
    info["total_videos"] = int(video_report["count"])
    info["splits"] = {"train": f"0:{episodes.num_rows}"}
    info["total_chunks"] = 1
    info["data_files_size_in_mb"] = total_file_size(out, "data/**/*.parquet") / 1_000_000
    info["video_files_size_in_mb"] = video_report["bytes"] / 1_000_000


def task_from_raw_dir(name: str) -> str | None:
    match = RAW_TASK_RE.match(name)
    if not match:
        return None
    obj = match.group("object").replace("_", " ")
    target = match.group("target").replace("_", " ")
    return f"pick up the {obj} and place it near the {target}"


def require_h5_dataset(h5: h5py.File, key: str) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Missing HDF5 dataset {key}")
    return np.asarray(h5[key], dtype=np.float32)


def load_raw_candidates(raw_root: Path) -> list[RawCandidate]:
    candidates: list[RawCandidate] = []
    for h5_path in sorted(raw_root.glob("*/data.hdf5")):
        with h5py.File(h5_path, "r") as h5:
            action = require_h5_dataset(h5, "data/demo_0/actions")
            joint_action = require_h5_dataset(h5, "data/demo_0/action_descriptions/joint_action")
            state = require_h5_dataset(h5, "data/demo_0/states/articulation/robot/joint_position")
            velocity = require_h5_dataset(h5, "data/demo_0/states/articulation/robot/joint_velocity")
            ee_position = require_h5_dataset(h5, "data/demo_0/ee_pose/position")
            ee_orientation = require_h5_dataset(h5, "data/demo_0/ee_pose/orientation")
        candidates.append(
            RawCandidate(
                path=h5_path,
                task=task_from_raw_dir(h5_path.parent.name),
                action=action,
                state=state,
                velocity=velocity,
                ee_position=ee_position,
                ee_orientation=ee_orientation,
                raw_action_joint_max_abs=float(np.max(np.abs(action - joint_action))),
            )
        )
    if not candidates:
        raise FileNotFoundError(f"No raw HDF5 files found under {raw_root}")
    return candidates


def episode_slices(episodes: pa.Table) -> list[tuple[int, int, int, str | None]]:
    rows = episodes.to_pylist()
    slices: list[tuple[int, int, int, str | None]] = []
    for row in rows:
        tasks = row.get("tasks") or []
        task = tasks[0] if tasks else None
        slices.append((int(row["episode_index"]), int(row["dataset_from_index"]), int(row["dataset_to_index"]), task))
    return slices


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def score_candidate(output_action: np.ndarray, output_state: np.ndarray, candidate: RawCandidate) -> tuple[float, int]:
    n = min(len(output_action), len(candidate.action))
    if n == 0:
        return float("inf"), abs(len(output_action) - len(candidate.action))
    action_mse = float(np.mean((output_action[:n] - candidate.action[:n]) ** 2))
    state_mse = float(np.mean((output_state[:n] - candidate.state[:n]) ** 2))
    len_diff = abs(len(output_action) - len(candidate.action))
    return action_mse + state_mse + len_diff, len_diff


def validate_raw_streams(
    out: Path,
    table: pa.Table,
    episodes: pa.Table,
    raw_root: Path,
    mode: str,
    sample_episodes: int,
    action_tol: float,
    state_tol: float,
) -> dict[str, Any]:
    if mode == "none":
        return {"enabled": False}
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    candidates = load_raw_candidates(raw_root)
    action = array_from_column(table, "action", np.float32)
    state = array_from_column(table, "observation.state", np.float32)
    velocity = array_from_column(table, "observation.velocity", np.float32)
    ee_position = array_from_column(table, "observation.ee_position", np.float32)
    ee_orientation = array_from_column(table, "observation.ee_orientation", np.float32)

    slices = episode_slices(episodes)
    if mode == "sample":
        slices = slices[:sample_episodes]

    matches: list[dict[str, Any]] = []
    max_action = 0.0
    max_state = 0.0
    max_velocity = 0.0
    max_ee_position = 0.0
    max_ee_orientation = 0.0
    task_mismatches: list[dict[str, Any]] = []
    for episode_index, start, end, episode_task in slices:
        out_action = action[start:end]
        out_state = state[start:end]
        ranked = sorted(
            ((score_candidate(out_action, out_state, cand), cand) for cand in candidates),
            key=lambda item: (item[0][0], item[0][1], str(item[1].path)),
        )
        best = ranked[0][1]
        n = len(out_action)
        if len(best.action) != n:
            raise ValueError(f"Episode {episode_index} best raw match length {len(best.action)} != output length {n}: {best.path}")

        action_diff = max_abs_diff(out_action, best.action)
        state_diff = max_abs_diff(out_state, best.state)
        velocity_diff = max_abs_diff(velocity[start:end], best.velocity)
        ee_pos_diff = max_abs_diff(ee_position[start:end], best.ee_position)
        ee_ori_diff = max_abs_diff(ee_orientation[start:end], best.ee_orientation)
        max_action = max(max_action, action_diff)
        max_state = max(max_state, state_diff)
        max_velocity = max(max_velocity, velocity_diff)
        max_ee_position = max(max_ee_position, ee_pos_diff)
        max_ee_orientation = max(max_ee_orientation, ee_ori_diff)
        if action_diff > action_tol:
            raise ValueError(f"Episode {episode_index} action max diff {action_diff} > {action_tol}: {best.path}")
        for label, value in (
            ("state", state_diff),
            ("velocity", velocity_diff),
            ("ee_position", ee_pos_diff),
            ("ee_orientation", ee_ori_diff),
        ):
            if value > state_tol:
                raise ValueError(f"Episode {episode_index} {label} max diff {value} > {state_tol}: {best.path}")
        if best.raw_action_joint_max_abs > action_tol:
            raise ValueError(f"Raw actions and joint_action differ by {best.raw_action_joint_max_abs}: {best.path}")
        if episode_task and best.task and episode_task != best.task:
            task_mismatches.append(
                {
                    "episode_index": episode_index,
                    "episode_task": episode_task,
                    "raw_task": best.task,
                    "raw_path": str(best.path),
                }
            )
        matches.append(
            {
                "episode_index": episode_index,
                "raw_path": str(best.path),
                "task": episode_task,
                "action_max_abs": action_diff,
                "state_max_abs": state_diff,
                "velocity_max_abs": velocity_diff,
                "ee_position_max_abs": ee_pos_diff,
                "ee_orientation_max_abs": ee_ori_diff,
            }
        )

    if task_mismatches:
        raise ValueError(f"Task/raw mismatches: {task_mismatches[:3]}")
    return {
        "enabled": True,
        "mode": mode,
        "raw_root": str(raw_root),
        "checked_episodes": len(slices),
        "available_raw_files": len(candidates),
        "max_action_abs": max_action,
        "max_state_abs": max_state,
        "max_velocity_abs": max_velocity,
        "max_ee_position_abs": max_ee_position,
        "max_ee_orientation_abs": max_ee_orientation,
        "matches_preview": matches[:10],
        "report_path": str(out / "joint_conversion_report.json"),
    }


def validate_core_dataset(out: Path, info: dict[str, Any], table: pa.Table, episodes: pa.Table, removed_fields: list[str]) -> dict[str, Any]:
    assert_feature_widths(table, info)
    if info["total_frames"] != table.num_rows:
        raise ValueError(f"info total_frames {info['total_frames']} != parquet rows {table.num_rows}")
    if info["total_episodes"] != episodes.num_rows:
        raise ValueError(f"info total_episodes {info['total_episodes']} != episode rows {episodes.num_rows}")
    lengths = np.asarray(episodes.column("length").combine_chunks().to_pylist(), dtype=np.int64)
    if int(lengths.sum()) != table.num_rows:
        raise ValueError(f"Sum episode lengths {int(lengths.sum())} != parquet rows {table.num_rows}")
    if removed_fields:
        lingering = [key for key in removed_fields if key in table.column_names or key in info.get("features", {})]
        if lingering:
            raise ValueError(f"Auxiliary action fields were not fully removed: {lingering}")

    indexes = array_from_column(table, "index", np.int64).reshape(-1)
    if not np.array_equal(indexes, np.arange(table.num_rows, dtype=np.int64)):
        raise ValueError("index column is not contiguous from 0")
    episode_index = array_from_column(table, "episode_index", np.int64).reshape(-1)
    frame_index = array_from_column(table, "frame_index", np.int64).reshape(-1)
    timestamp = array_from_column(table, "timestamp", np.float64).reshape(-1)
    done = table.column("next.done").combine_chunks().to_pylist() if "next.done" in table.column_names else None
    fps = float(info["fps"])
    for row in episodes.to_pylist():
        ep = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        expected_frames = np.arange(end - start, dtype=np.int64)
        if not np.all(episode_index[start:end] == ep):
            raise ValueError(f"episode_index mismatch in episode {ep}")
        if not np.array_equal(frame_index[start:end], expected_frames):
            raise ValueError(f"frame_index mismatch in episode {ep}")
        if not np.allclose(timestamp[start:end], expected_frames / fps, atol=2e-4):
            raise ValueError(f"timestamp/fps mismatch in episode {ep}")
        if done is not None:
            ep_done = done[start:end]
            if any(ep_done[:-1]) or ep_done[-1] is not True:
                raise ValueError(f"next.done mismatch in episode {ep}")

    action = array_from_column(table, "action", np.float32)
    if not np.isfinite(action).all():
        raise ValueError("Non-finite action values")
    gripper = action[:, 6]
    if np.min(gripper) < -1e-6 or np.max(gripper) > 1.0 + 1e-6:
        raise ValueError(f"Gripper channel outside [0, 1]: min={float(np.min(gripper))} max={float(np.max(gripper))}")

    return {
        "root": str(out),
        "episodes": int(episodes.num_rows),
        "frames": int(table.num_rows),
        "fps": fps,
        "columns": table.column_names,
        "removed_aux_action_fields": removed_fields,
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    out = args.out.resolve()
    if not (src / "meta" / "info.json").is_file():
        raise FileNotFoundError(src / "meta" / "info.json")

    info = load_json(src / "meta" / "info.json")
    src_stats = load_json(src / "meta" / "stats.json")
    data = pq.read_table(source_data_path(src, info))
    episodes = pq.read_table(source_episode_path(src))
    data, episodes = filter_first_episodes(data, episodes, args.max_episodes)
    info = copy.deepcopy(info)
    assert_feature_widths(data, info)
    data, removed_fields = drop_aux_action_columns(data, info, args.keep_aux_action_fields)

    prepare_output(out, args.overwrite)
    materialize_meta_files(src, out)
    dst_data_path(out, info).parent.mkdir(parents=True, exist_ok=True)
    dst_episode_path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, dst_data_path(out, info))
    pq.write_table(episodes, dst_episode_path(out))
    video_report = materialize_videos(src, out, info, episodes, args.link_mode)
    update_info_counts(info, out, data, episodes, video_report)
    write_json(out / "meta" / "info.json", info)
    stats = compute_numeric_stats(data, info, src_stats)
    write_json(out / "meta" / "stats.json", stats)

    core_report = validate_core_dataset(out, info, data, episodes, removed_fields)
    raw_report = validate_raw_streams(
        out=out,
        table=data,
        episodes=episodes,
        raw_root=args.raw_root.resolve(),
        mode=args.raw_validate,
        sample_episodes=args.raw_sample_episodes,
        action_tol=args.tolerance,
        state_tol=args.state_tolerance,
    )
    report = {
        "ok": True,
        "source_root": str(src),
        "output_root": str(out),
        "max_episodes": args.max_episodes,
        "keep_aux_action_fields": args.keep_aux_action_fields,
        "core": core_report,
        "video": video_report,
        "raw_validation": raw_report,
    }
    write_json(out / "joint_conversion_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
