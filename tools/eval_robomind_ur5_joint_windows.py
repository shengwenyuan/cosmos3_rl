#!/usr/bin/env python3
"""Evaluate a UR5 joint policy on exact, moving RoboMIND training windows.

The test deliberately uses the same raw dataset adapter and the same
``RobolabPolicyService.infer`` path as online serving.  It writes the selected
conditioning image/state and ground-truth action chunk before model loading so
the exact window can also be replayed by RoboLab.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset import (
    UR5SingleLeRobotDataset,
    _bind_ur5_joint_manifest_sources,
)
from cosmos_framework.data.generator.action.policy_schema import (
    find_action_policy_manifest,
    load_action_policy_manifest,
)

# One real ``pick up the banana`` episode: frame 13 stresses arm motion, while
# frame 57 includes the gripper-closing transition.  Both are part of the exact
# 22,044-episode training mirror.
DEFAULT_WINDOWS = ("748:13", "748:57")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--dataset-source")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mlp_vepfs/share/swy/cosmos3-framework/lerobot/RoboMIND1-ur5"),
        help="Physical LeRobot root. This may override a stale historical path in the immutable run sidecar.",
    )
    parser.add_argument(
        "--window",
        action="append",
        metavar="EPISODE:FRAME",
        help="Exact valid window start. Repeat for multiple windows; defaults to two deterministic moving windows.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true", help="Write replay inputs without loading the model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_manifest(args: argparse.Namespace):
    manifest_path = args.policy_config or find_action_policy_manifest(args.checkpoint_path)
    if manifest_path is None:
        raise ValueError("No action_policy.yaml found; pass --policy-config explicitly")
    manifest = load_action_policy_manifest(manifest_path)
    source = manifest.resolve_dataset_source(args.dataset_source)
    if manifest.model_action.codec != "joint_position" or manifest.model_action_dim != 7:
        raise ValueError("This test requires a 7-D UR5 joint-position policy")
    return Path(manifest_path), manifest, source


def _parse_windows(values: list[str] | None) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for value in values or DEFAULT_WINDOWS:
        try:
            episode_text, frame_text = value.split(":", maxsplit=1)
            pair = (int(episode_text), int(frame_text))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid --window {value!r}; expected EPISODE:FRAME") from error
        if pair[0] < 0 or pair[1] < 0:
            raise ValueError(f"Window coordinates must be non-negative, got {pair}")
        result.append(pair)
    return result


def _flat_index(dataset: UR5SingleLeRobotDataset, episode: int, frame: int) -> tuple[int, int]:
    flat_start = 0
    for dataset_index, row_start, valid_length, episode_id in dataset._episode_records:
        if episode_id == episode:
            if not 0 <= frame < valid_length:
                raise ValueError(
                    f"Episode {episode} frame {frame} is not a valid {dataset.chunk_length}-step window; "
                    f"valid starts are 0..{valid_length - 1}"
                )
            return flat_start + frame, row_start + frame
        flat_start += valid_length
    raise ValueError(f"Episode {episode} is not present in the selected dataset source")


def _load_windows(args: argparse.Namespace, manifest, source) -> list[dict[str, Any]]:
    source_configs = [config for config in _bind_ur5_joint_manifest_sources(manifest) if config["name"] == source.name]
    if len(source_configs) != 1:
        raise ValueError(f"Expected exactly one bound source named {source.name!r}, got {len(source_configs)}")
    source_configs[0]["root"] = str(args.dataset_root)

    dataset = UR5SingleLeRobotDataset(
        sources=source_configs,
        fps=float(manifest.policy_fps),
        chunk_length=int(manifest.chunk_size),
        split="full",
        split_val_ratio=0.0,
        mode="policy",
        action_normalization=None,
        video_backend="torchcodec",
    )
    loaded: list[dict[str, Any]] = []
    for episode, frame in _parse_windows(args.window):
        flat_index, parquet_row = _flat_index(dataset, episode, frame)
        sample = dataset[flat_index]
        action = torch.as_tensor(sample["action"]).detach().cpu().numpy().astype(np.float32, copy=False)
        video = torch.as_tensor(sample["video"]).detach().cpu().numpy()
        expected = (manifest.chunk_size + manifest.conditioning.state_rows, manifest.model_action_dim)
        if action.shape != expected:
            raise RuntimeError(f"Dataset action shape {action.shape} does not match manifest {expected}")
        if video.ndim != 4 or video.shape[1] != manifest.chunk_size + 1:
            raise RuntimeError(f"Dataset video has unexpected shape {video.shape}")
        image = np.transpose(video[:, 0], (1, 2, 0)).astype(np.uint8, copy=False)
        state = action[0]
        target = action[manifest.conditioning.history_rows :]
        caption = str(sample["ai_caption"])
        arm_hold_rmse = float(np.sqrt(np.mean(np.square(target[:, :6] - state[None, :6]))))
        loaded.append(
            {
                "episode": episode,
                "frame": frame,
                "flat_index": flat_index,
                "parquet_row": parquet_row,
                "caption": caption,
                "image": image,
                "state": state,
                "target": target,
                "arm_hold_rmse": arm_hold_rmse,
            }
        )
    return loaded


def _save_replay_inputs(output_dir: Path, windows: list[dict[str, Any]], manifest_path: Path, args) -> None:
    rows = []
    for window in windows:
        stem = f"episode_{window['episode']:05d}_frame_{window['frame']:04d}"
        bundle_path = output_dir / f"{stem}_input.npz"
        np.savez_compressed(
            bundle_path,
            image=window["image"],
            state=window["state"],
            target=window["target"],
            caption=np.asarray(window["caption"]),
            episode=np.asarray(window["episode"], dtype=np.int64),
            frame=np.asarray(window["frame"], dtype=np.int64),
            flat_index=np.asarray(window["flat_index"], dtype=np.int64),
            parquet_row=np.asarray(window["parquet_row"], dtype=np.int64),
        )
        rows.append(
            {key: window[key] for key in ("episode", "frame", "flat_index", "parquet_row", "caption", "arm_hold_rmse")}
            | {"input_bundle": str(bundle_path)}
        )
    _write_json(
        output_dir / "inputs.json",
        {
            "checkpoint_path": args.checkpoint_path,
            "config_file": args.config_file,
            "manifest_path": manifest_path,
            "dataset_root": args.dataset_root,
            "windows": rows,
        },
    )


def _metrics(state: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    arm_error = prediction[:, :6] - target[:, :6]
    hold_error = state[None, :6] - target[:, :6]
    joint_rmse = float(np.sqrt(np.mean(np.square(arm_error))))
    hold_rmse = float(np.sqrt(np.mean(np.square(hold_error))))
    target_delta = target[-1, :6] - state[:6]
    prediction_delta = prediction[-1, :6] - state[:6]
    denom = float(np.linalg.norm(target_delta) * np.linalg.norm(prediction_delta))
    endpoint_direction_cosine = float(np.dot(target_delta, prediction_delta) / denom) if denom > 1e-12 else None
    gt_closed = target[:, 6] >= 0.5
    pred_closed = prediction[:, 6] >= 0.5
    return {
        "joint_mae_rad": float(np.mean(np.abs(arm_error))),
        "joint_rmse_rad": joint_rmse,
        "joint_max_abs_rad": float(np.max(np.abs(arm_error))),
        "joint_per_axis_mae_rad": np.mean(np.abs(arm_error), axis=0),
        "endpoint_joint_l2_rad": float(np.linalg.norm(arm_error[-1])),
        "endpoint_direction_cosine": endpoint_direction_cosine,
        "hold_joint_rmse_rad": hold_rmse,
        "rmse_improvement_vs_hold_fraction": (1.0 - joint_rmse / hold_rmse) if hold_rmse > 1e-12 else None,
        "first_prediction_jump_l2_rad": float(np.linalg.norm(prediction[0, :6] - state[:6])),
        "first_target_jump_l2_rad": float(np.linalg.norm(target[0, :6] - state[:6])),
        "gripper_mae": float(np.mean(np.abs(prediction[:, 6] - target[:, 6]))),
        "gripper_binary_accuracy": float(np.mean(pred_closed == gt_closed)),
        "target_gripper_range": [float(target[:, 6].min()), float(target[:, 6].max())],
        "prediction_gripper_range": [float(prediction[:, 6].min()), float(prediction[:, 6].max())],
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path, manifest, source = _resolve_manifest(args)
    started = time.perf_counter()
    windows = _load_windows(args, manifest, source)
    _save_replay_inputs(args.output_dir, windows, manifest_path, args)
    print(f"[robomind-window-eval] replay inputs ready at {args.output_dir}", flush=True)
    if args.data_only:
        return

    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    service = RobolabPolicyService(
        RobolabServerArgs(
            checkpoint_path=args.checkpoint_path,
            config_file=args.config_file,
            policy_config=args.policy_config,
            dataset_source=args.dataset_source,
            allow_dcp_checkpoint=True,
            output_dir=args.output_dir / "omni",
            seed=args.seed,
            deterministic_seed=True,
            guidance=args.guidance,
            num_steps=args.num_steps,
            shift=args.shift,
        )
    )

    results = []
    for window in windows:
        obs = {
            "observation/image": window["image"],
            "observation/joint_position": window["state"][:6],
            "observation/gripper_position": window["state"][6:7],
            "prompt": window["caption"],
        }
        infer_started = time.perf_counter()
        prediction = np.asarray(service.infer(obs)["action"], dtype=np.float32)
        infer_seconds = time.perf_counter() - infer_started
        if prediction.shape != window["target"].shape:
            raise RuntimeError(f"Prediction shape {prediction.shape} does not match target {window['target'].shape}")
        metrics = _metrics(window["state"], window["target"], prediction)
        stem = f"episode_{window['episode']:05d}_frame_{window['frame']:04d}"
        np.savez_compressed(
            args.output_dir / f"{stem}_prediction.npz",
            state=window["state"],
            target=window["target"],
            prediction=prediction,
        )
        row = {
            "episode": window["episode"],
            "frame": window["frame"],
            "flat_index": window["flat_index"],
            "parquet_row": window["parquet_row"],
            "caption": window["caption"],
            "inference_seconds": infer_seconds,
            **metrics,
        }
        _write_json(args.output_dir / f"{stem}_metrics.json", row)
        results.append(row)
        print(
            f"[robomind-window-eval] episode={window['episode']} frame={window['frame']} "
            f"joint_rmse={metrics['joint_rmse_rad']:.6f} hold_rmse={metrics['hold_joint_rmse_rad']:.6f} "
            f"gripper_mae={metrics['gripper_mae']:.6f}",
            flush=True,
        )

    aggregate = {
        "num_windows": len(results),
        "mean_joint_rmse_rad": float(np.mean([row["joint_rmse_rad"] for row in results])),
        "mean_hold_joint_rmse_rad": float(np.mean([row["hold_joint_rmse_rad"] for row in results])),
        "mean_gripper_mae": float(np.mean([row["gripper_mae"] for row in results])),
        "total_seconds": time.perf_counter() - started,
        "windows": results,
    }
    _write_json(args.output_dir / "summary.json", aggregate)
    print(f"[robomind-window-eval] summary={args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
