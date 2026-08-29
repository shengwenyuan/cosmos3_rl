#!/usr/bin/env python3
"""Fail-fast preflight for RH20T cfg4 UR5 joint Edge training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import tomllib

EXPECTED_LAYOUT = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper_close_fraction",
]
EXPECTED_CAMERAS = {
    "primary": "observation.images.exterior_1",
    "aux_left": "observation.images.exterior_2",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _manifest_windows(path: Path) -> list[int]:
    windows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            count = record.get("window_count_c32")
            _require(isinstance(count, int) and count > 0, f"Invalid window_count_c32 at line {line_number}")
            windows.append(count)
    return windows


def validate(args: argparse.Namespace) -> dict[str, Any]:
    with args.toml.open("rb") as stream:
        config = tomllib.load(stream)
    info = _load_json(args.dataset_root / "meta/info.json")
    stats = _load_json(args.action_stats)
    window_counts = _manifest_windows(args.train_manifest)

    _require(info.get("fps") == 15, "Dataset fps must be 15")
    _require(info.get("total_episodes") == args.expected_episodes, "Unexpected dataset episode count")
    _require(info.get("total_frames") == args.expected_frames, "Unexpected dataset frame count")
    _require(len(window_counts) == args.expected_episodes, "Train manifest must map one-to-one to episodes")
    _require(sum(window_counts) == args.expected_windows, "Unexpected train c32 window count")

    features = info.get("features", {})
    _require(features.get("action", {}).get("names", {}).get("motors") == EXPECTED_LAYOUT, "Action layout drift")
    _require(features.get("observation.state.joint", {}).get("shape") == [6], "Joint state must be 6-D")
    _require(features.get("observation.state.gripper", {}).get("shape") == [1], "Gripper state must be 1-D")
    for feature in EXPECTED_CAMERAS.values():
        camera = features.get(feature, {})
        _require(camera.get("dtype") == "video", f"Missing video feature {feature}")
        _require(camera.get("shape") == [360, 640, 3], f"Unexpected camera shape for {feature}")

    digest = hashlib.sha256(args.action_stats.read_bytes()).hexdigest()
    policy = config["action_policy"]
    normalization = policy["normalization"]
    _require(digest == normalization.get("sha256"), "Action stats SHA256 does not match TOML")
    _require(normalization.get("kind") == "quantile", "Normalization must be per-dimension quantile")
    _require(normalization.get("apply_forward_clamp") is False, "Baseline must not clamp training targets")
    metadata = stats.get("metadata", {})
    _require(metadata.get("split") == "train", "Action stats must be train-only")
    _require(metadata.get("action_layout") == EXPECTED_LAYOUT, "Action stats layout drift")
    _require(metadata.get("num_action_rows") == args.expected_frames, "Action stats row count drift")

    experiment = config["job"].get("experiment")
    _require(
        experiment
        in {"action_policy_ur5_single_joint_edge", "action_policy_ur5_single_joint_edge_overfit"},
        "Wrong experiment",
    )
    _require(policy.get("policy_fps") == 15 and policy.get("chunk_size") == 32, "Policy must be 15 Hz/c32")
    observation = policy["observation"]
    _require(observation.get("layout_id") == "vertical_pair", "Observation must use vertical_pair")
    _require(observation.get("view_shape_hw") == [360, 640], "View shape must be 360x640")
    _require(observation.get("canvas_shape_hw") == [720, 640], "Canvas must be 720x640")
    _require(observation.get("video_subsample") == 2, "video_subsample must be 2")
    _require(observation.get("missing_view_policy") == "error", "Both real views are mandatory")
    _require(policy["conditioning"] == {
        "state_rows": 1,
        "history_rows": 1,
        "source": "current_state",
        "timing": policy["conditioning"]["timing"],
    }, "Policy must use one current-state conditioning row")
    _require(policy["model_action"].get("codec") == "joint_position", "Model action must be joint_position")
    _require(policy["model_action"].get("representation") == "absolute", "Model action must be absolute")

    sources = policy.get("datasets", [])
    _require(len(sources) == 1, "Exactly one RH20T source is required")
    source = sources[0]
    _require(source.get("root") == "${oc.env:RH20T_TRAIN_ROOT}", "Dataset root must be environment-bound")
    _require(source.get("condition_source") == "observation_state_t0", "State condition source drift")
    _require(source.get("action_features") == ["action"], "Action feature drift")
    _require(
        source.get("state_features") == ["observation.state.joint", "observation.state.gripper"],
        "State feature drift",
    )
    _require(source.get("camera_features") == EXPECTED_CAMERAS, "Camera identity/order drift")
    _require(source.get("action_layout") == EXPECTED_LAYOUT, "Source action layout drift")

    parallel = config["model"]["parallelism"]
    configured_world = (
        parallel.get("data_parallel_shard_degree", 1)
        * parallel.get("data_parallel_replicate_degree", 1)
        * parallel.get("context_parallel_shard_degree", 1)
        * parallel.get("cfg_parallel_shard_degree", 1)
    )
    _require(configured_world == args.world_size, "TOML parallelism does not match requested world size")
    per_rank = config["dataloader_train"].get("max_samples_per_batch")
    accumulation = config["trainer"].get("grad_accum_iter")
    _require(per_rank == 1, "Baseline requires one sample per rank")
    _require(isinstance(accumulation, int) and accumulation > 0, "Invalid gradient accumulation")
    global_batch = args.world_size * per_rank * accumulation

    episode_indices = source.get("episode_indices")
    sample_stride = 8 if experiment.endswith("_overfit") else 1
    _require(
        (episode_indices is not None) == experiment.endswith("_overfit"),
        "Only the overfit experiment may select episode_indices",
    )
    if episode_indices is None:
        selected_windows = sum(math.ceil(count / sample_stride) for count in window_counts)
        selected_episodes = args.expected_episodes
    else:
        _require(episode_indices == sorted(set(episode_indices)), "Overfit episode_indices must be sorted/unique")
        _require(all(0 <= index < len(window_counts) for index in episode_indices), "Overfit episode index out of range")
        selected_windows = sum(math.ceil(window_counts[index] / sample_stride) for index in episode_indices)
        selected_episodes = len(episode_indices)

    max_iter = config["trainer"].get("max_iter")
    _require(config["scheduler"].get("cycle_lengths") == [max_iter], "Scheduler cycle must equal max_iter")
    _require(config["scheduler"].get("f_min") == [0.1], "Scheduler LR floor must be 0.1")
    exposures = global_batch * max_iter
    return {
        "status": "pass",
        "toml": str(args.toml),
        "dataset": {
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "windows_c32": sum(window_counts),
        },
        "selection": {"episodes": selected_episodes, "windows_c32": selected_windows},
        "training": {
            "world_size": args.world_size,
            "per_rank_batch": per_rank,
            "grad_accum_iter": accumulation,
            "global_batch": global_batch,
            "sample_stride": sample_stride,
            "max_iter": max_iter,
            "sample_exposures": exposures,
            "window_passes": round(exposures / selected_windows, 4),
            "steps_per_window_pass": math.ceil(selected_windows / global_batch),
        },
        "action_stats_sha256": digest,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=1776)
    parser.add_argument("--expected-frames", type=int, default=1_255_431)
    parser.add_argument("--expected-windows", type=int, default=1_198_599)
    return parser


def main(argv: list[str] | None = None) -> int:
    report = validate(_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
