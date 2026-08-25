#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline distribution gate for a DROID panda_link8 EEF checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HORIZONS = (1, 8, 16, 32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert two matrix columns to a proper rotation matrix."""
    values = np.asarray(rot6d, dtype=np.float64)
    first = values[..., :3]
    second = values[..., 3:6]
    first /= np.clip(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12, None)
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second /= np.clip(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12, None)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((values[..., :, 0], values[..., :, 1]), axis=-1)


def anchored_pose_from_action(action: np.ndarray) -> tuple[np.ndarray, Rotation]:
    values = np.asarray(action, dtype=np.float64)
    return values[..., :3], Rotation.from_matrix(rot6d_to_matrix(values[..., 3:9]))


def absolute_pose_to_anchored_action(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    initial_position: np.ndarray,
    initial_quaternion_xyzw: np.ndarray,
    gripper_open_fraction: np.ndarray,
) -> np.ndarray:
    initial_rotation = Rotation.from_quat(initial_quaternion_xyzw)
    rotations = Rotation.from_quat(quaternions_xyzw)
    relative_positions = initial_rotation.inv().apply(np.asarray(positions) - initial_position)
    relative_rotations = initial_rotation.inv() * rotations
    return np.concatenate(
        (
            relative_positions,
            matrix_to_rot6d(relative_rotations.as_matrix()),
            np.asarray(gripper_open_fraction, dtype=np.float64).reshape(-1, 1),
        ),
        axis=-1,
    )


def adjacent_motion(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions, rotations = anchored_pose_from_action(action)
    origin_position = np.zeros((1, 3), dtype=np.float64)
    origin_rotation = Rotation.identity(1)
    previous_positions = np.concatenate((origin_position, positions[:-1]), axis=0)
    previous_rotations = Rotation.concatenate((origin_rotation, rotations[:-1]))
    translation = previous_rotations.inv().apply(positions - previous_positions)
    rotation = previous_rotations.inv() * rotations
    return np.linalg.norm(translation, axis=-1), rotation.magnitude()


def horizon_motion(action: np.ndarray, horizons: Sequence[int] = HORIZONS) -> dict[int, tuple[float, float]]:
    positions, rotations = anchored_pose_from_action(action)
    return {
        int(horizon): (
            float(np.linalg.norm(positions[horizon - 1])),
            float(rotations[horizon - 1].magnitude()),
        )
        for horizon in horizons
    }


def percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max(initial=0.0)),
    }


def scaled_percentiles(values: np.ndarray, scale: float) -> dict[str, float]:
    return {key: value * scale for key, value in percentiles(values).items()}


def load_episode_shards(dataset_root: Path) -> dict[int, tuple[int, int]]:
    import pyarrow.dataset as arrow_dataset

    table = arrow_dataset.dataset(dataset_root / "meta/episodes", format="parquet").to_table(
        columns=["episode_index", "data/chunk_index", "data/file_index"]
    )
    episodes = table.column("episode_index").to_numpy()
    chunks = table.column("data/chunk_index").to_numpy()
    files = table.column("data/file_index").to_numpy()
    return {
        int(episode): (int(chunk), int(file))
        for episode, chunk, file in zip(episodes, chunks, files, strict=True)
    }


def select_indices(
    episode_records: Sequence[tuple[int, int, int, int]],
    episode_cum_ends: Sequence[int],
    episode_shards: dict[int, tuple[int, int]],
    *,
    num_samples: int,
    num_shards: int,
    seed: int,
) -> tuple[list[int], list[tuple[int, int]], list[int]]:
    grouped: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    previous_end = 0
    for record, end in zip(episode_records, episode_cum_ends, strict=True):
        _, _, length, episode_index = record
        shard = episode_shards[int(episode_index)]
        grouped[shard].append((previous_end, int(length), int(episode_index)))
        previous_end = int(end)
    available = sorted(grouped)
    if num_shards > len(available):
        raise ValueError(f"Requested {num_shards} shards, but only {len(available)} contain selected windows")

    rng = np.random.default_rng(seed)
    chosen = [available[index] for index in sorted(rng.choice(len(available), num_shards, replace=False))]
    indices: list[int] = []
    episodes: list[int] = []
    used: set[int] = set()
    for sample_index in range(num_samples):
        shard = chosen[sample_index % len(chosen)]
        spans = grouped[shard]
        for _ in range(100):
            start, length, episode = spans[int(rng.integers(len(spans)))]
            index = start + int(rng.integers(length))
            if index not in used:
                break
        else:
            raise RuntimeError(f"Could not choose a unique window from shard {shard}")
        used.add(index)
        indices.append(index)
        episodes.append(episode)
    return indices, chosen, episodes


def build_dataset(args: argparse.Namespace):
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
    from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset_config import (
        COSMOS3_DROID_SUCCESS_PROFILE,
        IMAGE_FEATURES,
    )

    dataset = DROIDLeRobotDataset(
        root=str(args.dataset_root),
        fps=15.0,
        chunk_length=32,
        split="train",
        action_normalization=None,
        viewpoint="concat_view",
        action_space="ee_pose_delta",
        use_state=True,
        training_manifest_path=str(args.training_manifest),
        pose_smoothing_window=5,
        dataset_profile=COSMOS3_DROID_SUCCESS_PROFILE,
        use_image_augmentation=False,
    )
    for image_key in IMAGE_FEATURES[COSMOS3_DROID_SUCCESS_PROFILE].values():
        # LeRobot squeezes a one-frame video query to [C,H,W]. Request T0/T1
        # so the standard compositor keeps [T,C,H,W], then use T0 below.
        dataset._delta_timestamps[image_key] = [0.0, dataset._dt]
    return dataset


def build_service(args: argparse.Namespace):
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    return RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=str(args.checkpoint),
            allow_dcp_checkpoint=True,
            experiment="action_policy_droid_eef_edge",
            config_file="cosmos_framework/configs/base/config.py",
            policy_config=args.policy_config,
            output_dir=args.output_dir / "model_runtime",
            seed=args.seed,
            deterministic_seed=False,
            guidance=3.0,
            num_steps=4,
            shift=5.0,
        )
    )


def initial_pose(initial_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(initial_action[:3], dtype=np.float64)
    quaternion = Rotation.from_matrix(rot6d_to_matrix(initial_action[3:9])).as_quat()
    return position, quaternion


def sample_observation(sample: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    action = np.asarray(sample["action"], dtype=np.float64)
    if action.shape != (33, 10):
        raise ValueError(f"Expected initial state plus 32 actions, got {action.shape}")
    position, quaternion = initial_pose(action[0])
    image = sample["video"][:, 0].permute(1, 2, 0).cpu().numpy()
    observation = {
        "prompt": sample["ai_caption"],
        "observation/image": image,
        "observation/eef_pose": np.concatenate((position, quaternion))[None].astype(np.float32),
        "observation/gripper_position": np.asarray([[1.0 - action[0, 9]]], dtype=np.float32),
    }
    return observation, action[1:]


def summarize(
    predicted_actions: np.ndarray,
    target_actions: np.ndarray,
    stats: dict[str, Any],
    *,
    ratio_limit: float,
    outlier_rate_limit: float,
) -> dict[str, Any]:
    predicted_horizon = {horizon: {"translation": [], "rotation": []} for horizon in HORIZONS}
    target_horizon = {horizon: {"translation": [], "rotation": []} for horizon in HORIZONS}
    for predicted, target in zip(predicted_actions, target_actions, strict=True):
        for horizon, (translation, rotation) in horizon_motion(predicted).items():
            predicted_horizon[horizon]["translation"].append(translation)
            predicted_horizon[horizon]["rotation"].append(rotation)
        for horizon, (translation, rotation) in horizon_motion(target).items():
            target_horizon[horizon]["translation"].append(translation)
            target_horizon[horizon]["rotation"].append(rotation)

    horizon_report: dict[str, Any] = {}
    ratio_failures: list[str] = []
    for horizon in HORIZONS:
        entry: dict[str, Any] = {}
        for name, scale in (("translation_cm", 100.0), ("rotation_deg", 180.0 / math.pi)):
            source = "translation" if name.startswith("translation") else "rotation"
            predicted = scaled_percentiles(np.asarray(predicted_horizon[horizon][source]), scale)
            target = scaled_percentiles(np.asarray(target_horizon[horizon][source]), scale)
            ratios = {
                quantile: predicted[quantile] / max(target[quantile], 1e-12)
                for quantile in ("p90", "p99")
            }
            if any(value > ratio_limit for value in ratios.values()):
                ratio_failures.append(f"h{horizon}.{name}")
            entry[name] = {"predicted": predicted, "target": target, "ratio": ratios}
        horizon_report[str(horizon)] = entry

    predicted_step_t, predicted_step_r = zip(*(adjacent_motion(action) for action in predicted_actions), strict=True)
    target_step_t, target_step_r = zip(*(adjacent_motion(action) for action in target_actions), strict=True)
    adjacent_report: dict[str, Any] = {}
    for name, predicted, target, scale in (
        ("translation_cm", np.concatenate(predicted_step_t), np.concatenate(target_step_t), 100.0),
        ("rotation_deg", np.concatenate(predicted_step_r), np.concatenate(target_step_r), 180.0 / math.pi),
    ):
        predicted_summary = scaled_percentiles(predicted, scale)
        target_summary = scaled_percentiles(target, scale)
        ratios = {
            quantile: predicted_summary[quantile] / max(target_summary[quantile], 1e-12)
            for quantile in ("p90", "p99")
        }
        if any(value > ratio_limit for value in ratios.values()):
            ratio_failures.append(f"adjacent.{name}")
        adjacent_report[name] = {"predicted": predicted_summary, "target": target_summary, "ratio": ratios}

    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    outside = (predicted_actions < q01) | (predicted_actions > q99)
    outside_rate = float(outside.mean())
    channel_outside_rate = outside.mean(axis=(0, 1)).tolist()
    offset = (q01 + q99) / 2.0
    zero_translation, zero_rotation = horizon_motion(offset[None], horizons=(1,))[1]
    failures = list(ratio_failures)
    if outside_rate > outlier_rate_limit:
        failures.append("raw_action.outside_q01_q99_rate")
    return {
        "horizons": horizon_report,
        "adjacent": adjacent_report,
        "raw_action": {
            "outside_q01_q99_rate": outside_rate,
            "outside_q01_q99_rate_by_channel": channel_outside_rate,
            "limit": outlier_rate_limit,
        },
        "model_zero_denormalized": {
            "translation_cm": zero_translation * 100.0,
            "rotation_deg": zero_rotation * 180.0 / math.pi,
            "raw_action": offset.tolist(),
        },
        "ratio_limit": ratio_limit,
        "failures": failures,
        "passed": not failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    service = build_service(args)
    dataset = build_dataset(args)
    episode_shards = load_episode_shards(args.dataset_root)
    indices, shards, episodes = select_indices(
        dataset._episode_records,
        dataset._episode_cum_ends,
        episode_shards,
        num_samples=args.num_samples,
        num_shards=args.num_shards,
        seed=args.seed,
    )
    predicted_actions: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    sample_records: list[dict[str, Any]] = []
    for order, (index, episode) in enumerate(zip(indices, episodes, strict=True), start=1):
        sample = dataset[index]
        observation, target = sample_observation(sample)
        output = service.infer(observation)
        position, quaternion = initial_pose(np.asarray(sample["action"])[0])
        wire = np.asarray(output["action"], dtype=np.float64)
        predicted = absolute_pose_to_anchored_action(
            wire[:, :3], wire[:, 3:7], position, quaternion, 1.0 - wire[:, 7]
        )
        predicted_actions.append(predicted)
        target_actions.append(target)
        sample_records.append(
            {
                "order": order,
                "dataset_index": index,
                "episode_index": episode,
                "data_shard": episode_shards[episode],
                "prompt": observation["prompt"],
                "predicted": {str(key): value for key, value in horizon_motion(predicted).items()},
                "target": {str(key): value for key, value in horizon_motion(target).items()},
            }
        )
        print(f"offline gate: {order}/{len(indices)}", flush=True)

    with args.stats.open() as handle:
        stats = json.load(handle)["global_raw"]
    report = summarize(
        np.stack(predicted_actions),
        np.stack(target_actions),
        stats,
        ratio_limit=args.ratio_limit,
        outlier_rate_limit=args.outlier_rate_limit,
    )
    report["sampling"] = {
        "num_samples": len(indices),
        "num_episodes": len(set(episodes)),
        "num_data_shards": len(set(shards)),
        "data_shards": shards,
        "seed": args.seed,
    }
    report["checkpoint"] = str(args.checkpoint)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "samples.jsonl").open("w") as handle:
        for record in sample_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print("DROID_EEF_OFFLINE_GATE=" + json.dumps(report, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=49)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio-limit", type=float, default=2.0)
    parser.add_argument("--outlier-rate-limit", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
