#!/usr/bin/env python3
"""Offline safety/scale gate for the frozen RH20T cfg4 UR5 joint policy."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RUN = Path(
    "/mnt/pfs/swy/cosmos3/runs/cosmos3_action_rh20t_ur5_joint/"
    "cosmos3_action_rh20t_ur5_joint/stage2_c32/rh20t_cfg4_ur5_joint_edge_64gpu_10k"
)
JOINT_LOWER = np.array([-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi])
JOINT_UPPER = -JOINT_LOWER
DEFAULT_GRIPPER_TOLERANCE = 0.05


def select_indices(length: int, count: int, seed: int) -> list[int]:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(length, size=min(count, length), replace=False))


def classify_action_window(action: np.ndarray) -> str:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 7 or len(action) < 2:
        raise ValueError(f"expected action window [T,7], got {action.shape}")
    gripper_event = bool(np.any(np.abs(np.diff(action[:, 6])) > 0.5))
    max_joint_step = float(np.abs(np.diff(action[:, :6], axis=0)).max())
    if gripper_event:
        return "gripper_event"
    if max_joint_step <= 0.003:
        return "low_motion"
    return "motion"


def select_balanced_indices(dataset: Any, count: int, seed: int, candidate_factor: int = 64) -> list[int]:
    candidates = select_indices(len(dataset), min(len(dataset), count * candidate_factor), seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index in candidates:
        groups[classify_action_window(_as_numpy(dataset[index]["action"]))].append(index)

    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    quotas = {"low_motion": count // 4, "gripper_event": count // 4, "motion": count - 2 * (count // 4)}
    selected: list[int] = []
    for group, quota in quotas.items():
        selected.extend(groups[group][:quota])
    if len(selected) < count:
        remainder = [index for index in candidates if index not in set(selected)]
        selected.extend(remainder[: count - len(selected)])
    return sorted(selected)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max(initial=0.0)),
    }


def _motion_summary(current: np.ndarray, chunks: np.ndarray, fps: int, execute_horizon: int) -> dict[str, Any]:
    horizon = min(execute_horizon, chunks.shape[1])
    positions = np.concatenate((current[:, None, :6], chunks[:, :horizon, :6]), axis=1)
    step = np.diff(positions, axis=1)
    velocity = step * fps
    acceleration = np.diff(velocity, axis=1) * fps
    return {
        "step_abs_rad": _quantiles(np.abs(step)),
        "velocity_abs_rad_s": _quantiles(np.abs(velocity)),
        "acceleration_abs_rad_s2": _quantiles(np.abs(acceleration)),
        "first_step_l2_rad": _quantiles(np.linalg.norm(step[:, 0], axis=-1)),
    }


def evaluate_arrays(
    current: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    fps: int = 15,
    execute_horizon: int = 8,
    max_velocity_rad_s: float = 3.2,
    max_acceleration_rad_s2: float = 20.0,
    max_p99_scale_ratio: float = 3.0,
    gripper_tolerance: float = DEFAULT_GRIPPER_TOLERANCE,
) -> dict[str, Any]:
    current = np.asarray(current, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if current.ndim != 2 or current.shape[1] != 7:
        raise ValueError(f"current must have shape [N,7], got {current.shape}")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != len(current):
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.shape[2] != 7:
        raise ValueError(f"prediction must be 7-D, got {prediction.shape}")
    if gripper_tolerance < 0.0:
        raise ValueError("gripper_tolerance must be non-negative")

    finite = bool(np.isfinite(prediction).all())
    joint = prediction[..., :6]
    limit_violations = int(np.count_nonzero((joint < JOINT_LOWER) | (joint > JOINT_UPPER)))
    gripper = prediction[..., 6]
    gripper_nominal_violations = int(np.count_nonzero((gripper < 0.0) | (gripper > 1.0)))
    gripper_violations = int(np.count_nonzero((gripper < -gripper_tolerance) | (gripper > 1.0 + gripper_tolerance)))
    predicted_motion = _motion_summary(current, prediction, fps, execute_horizon)
    target_motion = _motion_summary(current, target, fps, execute_horizon)
    target_p99 = target_motion["step_abs_rad"]["p99"]
    p99_scale_ratio = predicted_motion["step_abs_rad"]["p99"] / max(target_p99, 1e-6)
    failures: list[str] = []
    if not finite:
        failures.append("non_finite_action")
    if limit_violations:
        failures.append("joint_limit_violation")
    if gripper_violations:
        failures.append("gripper_range_violation")
    if predicted_motion["velocity_abs_rad_s"]["max"] > max_velocity_rad_s:
        failures.append("velocity_violation")
    if predicted_motion["acceleration_abs_rad_s2"]["max"] > max_acceleration_rad_s2:
        failures.append("acceleration_violation")
    if p99_scale_ratio > max_p99_scale_ratio:
        failures.append("motion_scale_ratio")

    return {
        "passed": not failures,
        "failures": failures,
        "counts": {
            "windows": int(len(current)),
            "joint_limit_violations": limit_violations,
            "gripper_nominal_range_violations": gripper_nominal_violations,
            "gripper_range_violations": gripper_violations,
        },
        "thresholds": {
            "execute_horizon": execute_horizon,
            "max_velocity_rad_s": max_velocity_rad_s,
            "max_acceleration_rad_s2": max_acceleration_rad_s2,
            "max_p99_scale_ratio": max_p99_scale_ratio,
            "gripper_tolerance": gripper_tolerance,
        },
        "gripper_prediction": {
            "min": float(gripper.min()),
            "max": float(gripper.max()),
        },
        "prediction": predicted_motion,
        "target": target_motion,
        "p99_step_scale_ratio": float(p99_scale_ratio),
        "arm_mae_rad": float(np.mean(np.abs(prediction[..., :6] - target[..., :6]))),
        "gripper_accuracy": float(np.mean((prediction[..., 6] > 0.5) == (target[..., 6] > 0.5))),
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_rgb_frame(video: Any) -> np.ndarray:
    video = _as_numpy(video)
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"expected dataset video [3,T,H,W], got {video.shape}")
    return np.ascontiguousarray(video[:, 0].transpose(1, 2, 0).astype(np.uint8, copy=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--samples", type=int, choices=(32, 256), default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--action-normalization-depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--joint-chunk-postprocessor", choices=("none", "triangular_3tap"), default="none")
    parser.add_argument("--gripper-tolerance", type=float, default=DEFAULT_GRIPPER_TOLERANCE)
    parser.add_argument("--execute-horizon", type=int, default=8)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    from cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset import (
        UR5SingleLeRobotDataset,
        _bind_ur5_joint_manifest_sources,
    )
    from cosmos_framework.data.generator.action.policy_schema import load_action_policy_manifest
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    manifest_path = args.run / "action_policy.yaml"
    manifest = load_action_policy_manifest(manifest_path)
    if manifest.profile_id != "rh20t_cfg4_ur5_joint_ext2_15hz_c32_v1":
        raise ValueError(f"unexpected policy profile {manifest.profile_id!r}")
    sources = _bind_ur5_joint_manifest_sources(manifest)
    common = dict(
        sources=sources,
        fps=manifest.policy_fps,
        chunk_length=manifest.chunk_size,
        video_subsample=manifest.observation.video_subsample,
        split="full",
        split_seed=42,
        split_val_ratio=0.0,
    )
    index_dataset = UR5SingleLeRobotDataset(**common, skip_video_loading=True)
    indices = select_balanced_indices(index_dataset, args.samples, args.seed)
    dataset = UR5SingleLeRobotDataset(**common, skip_video_loading=False, video_backend=args.video_backend)

    checkpoint = args.run / "checkpoints" / f"iter_{args.iteration:09d}" / "model"
    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=str(checkpoint),
            allow_dcp_checkpoint=True,
            config_file=str(args.run / "config.yaml"),
            policy_config=manifest_path,
            seed=args.seed,
            deterministic_seed=True,
            guidance=args.guidance,
            num_steps=args.num_steps,
            action_normalization_depth=args.action_normalization_depth,
            joint_chunk_postprocessor=args.joint_chunk_postprocessor,
        )
    )

    current_rows: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    categories: dict[str, int] = defaultdict(int)
    for ordinal, index in enumerate(indices, start=1):
        sample = dataset[index]
        action = _as_numpy(sample["action"]).astype(np.float32, copy=False)
        category = classify_action_window(action)
        categories[category] += 1
        current, target = action[0], action[1:]
        result = service.infer(
            {
                "prompt": str(sample["ai_caption"]),
                "observation/image": _first_rgb_frame(sample["video"]),
                "observation/joint_position": current[:6],
                "observation/gripper_position": current[6:7],
            }
        )
        prediction = np.asarray(result["action"], dtype=np.float32)
        raw_prediction = np.asarray(result.get("raw_action", result["action"]), dtype=np.float32)
        if prediction.shape != target.shape:
            raise ValueError(f"sample {index}: prediction shape {prediction.shape}, target {target.shape}")
        if raw_prediction.shape != target.shape:
            raise ValueError(f"sample {index}: raw prediction shape {raw_prediction.shape}, target {target.shape}")
        current_rows.append(current)
        predictions.append(prediction)
        raw_predictions.append(raw_prediction)
        targets.append(target)
        print(f"[{ordinal:03d}/{len(indices):03d}] index={index} category={category}", flush=True)

    current_array = np.stack(current_rows)
    prediction_array = np.stack(predictions)
    raw_prediction_array = np.stack(raw_predictions)
    target_array = np.stack(targets)
    report = evaluate_arrays(
        current_array,
        prediction_array,
        target_array,
        fps=manifest.policy_fps,
        execute_horizon=args.execute_horizon,
        gripper_tolerance=args.gripper_tolerance,
    )
    raw_prediction_gate = evaluate_arrays(
        current_array,
        raw_prediction_array,
        target_array,
        fps=manifest.policy_fps,
        execute_horizon=args.execute_horizon,
        gripper_tolerance=args.gripper_tolerance,
    )
    output = args.output or args.run / "evaluation" / f"iter_{args.iteration:09d}" / "offline_gate.json"
    samples_output = output.with_name(f"{output.stem}_samples.npz")
    report.update(
        {
            "profile_id": manifest.profile_id,
            "checkpoint": str(checkpoint.parent),
            "seed": args.seed,
            "num_steps": args.num_steps,
            "guidance": args.guidance,
            "action_normalization_depth": args.action_normalization_depth,
            "joint_chunk_postprocessor": args.joint_chunk_postprocessor,
            "raw_prediction_gate": raw_prediction_gate,
            "indices": indices,
            "categories": dict(sorted(categories.items())),
            "raw_samples": str(samples_output),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        samples_output,
        indices=np.asarray(indices, dtype=np.int64),
        current=current_array,
        prediction=prediction_array,
        raw_prediction=raw_prediction_array,
        target=target_array,
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
