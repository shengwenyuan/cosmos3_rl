# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fit and validate RH20T cfg4 joint→TCP constants against the official UR5 order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tools.rh20t.rh20t_io import (
    DEFAULT_DERIVED_ROOT,
    PRIMARY_SERIAL,
    atomic_write_json,
    load_joint_stream,
    load_tcp_base_stream,
    nearest_indices,
)

DEFAULT_MANIFEST = DEFAULT_DERIVED_ROOT / "manifests" / "train.jsonl"
DEFAULT_REPORT = DEFAULT_DERIVED_ROOT / "reports" / "joint_tcp_fk_gate.json"


def ur5_forward_kinematics(joints: np.ndarray) -> np.ndarray:
    """Standard UR5 DH FK in the official six-joint order."""

    a = (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0)
    d = (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823)
    alpha = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
    result = np.eye(4)
    for theta, link_a, link_d, link_alpha in zip(joints, a, d, alpha, strict=True):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(link_alpha), np.sin(link_alpha)
        result = result @ np.asarray(
            [
                [c, -s * ca, s * sa, link_a * c],
                [s, c * ca, -c * sa, link_a * s],
                [0.0, sa, ca, link_d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return result


def _transform(vector: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    result[:3, 3] = vector[:3]
    return result


def _tcp_matrix(pose: np.ndarray) -> np.ndarray:
    """RH20T stores quaternion as wxyz, unlike scipy's xyzw."""

    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_pairs(
    manifest: Path, episode_limit: int, samples_per_episode: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    records = _read_jsonl(manifest)
    source_paths: list[Path] = []
    seen: set[str] = set()
    for record in records:
        source_id = str(record["source_episode_id"])
        if source_id in seen:
            continue
        seen.add(source_id)
        source_paths.append(Path(record["source_path"]))
        if len(source_paths) >= episode_limit:
            break
    fk_rows: list[np.ndarray] = []
    tcp_rows: list[np.ndarray] = []
    episode_ids: list[str] = []
    for source in source_paths:
        joint_ts, joint = load_joint_stream(source / "transformed" / "joint.npy", PRIMARY_SERIAL)
        tcp_ts, tcp = load_tcp_base_stream(source / "transformed" / "tcp_base.npy", PRIMARY_SERIAL)
        indices, errors = nearest_indices(joint_ts, tcp_ts)
        valid = errors <= 100.0
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            continue
        selected = valid_indices[
            np.linspace(0, len(valid_indices) - 1, min(samples_per_episode, len(valid_indices))).astype(int)
        ]
        fk_rows.extend(ur5_forward_kinematics(joint[indices[index]]) for index in selected)
        tcp_rows.extend(_tcp_matrix(tcp[index]) for index in selected)
        episode_ids.extend([source.name] * len(selected))
    if not fk_rows:
        raise ValueError("No joint/TCP sample pairs found")
    return np.stack(fk_rows), np.stack(tcp_rows), episode_ids


def run_fk_gate(
    manifest: Path,
    *,
    episode_limit: int = 16,
    samples_per_episode: int = 24,
    position_p90_limit_m: float = 0.005,
    rotation_p90_limit_deg: float = 1.0,
) -> dict[str, Any]:
    fk, observed, episode_ids = _sample_pairs(manifest, episode_limit, samples_per_episode)

    def residual(vector: np.ndarray) -> np.ndarray:
        base = _transform(vector[:6])
        tool = _transform(vector[6:])
        predicted = base[None] @ fk @ tool[None]
        position = (predicted[:, :3, 3] - observed[:, :3, 3]).reshape(-1)
        rotation = (
            Rotation.from_matrix(np.swapaxes(predicted[:, :3, :3], 1, 2) @ observed[:, :3, :3]).as_rotvec().reshape(-1)
        )
        return np.concatenate((position, rotation * 0.1))

    initial = np.zeros(12)
    initial_base = observed[0] @ np.linalg.inv(fk[0])
    initial[:3] = initial_base[:3, 3]
    initial[3:6] = Rotation.from_matrix(initial_base[:3, :3]).as_rotvec()
    fit = least_squares(residual, initial, max_nfev=5000)
    base, tool = _transform(fit.x[:6]), _transform(fit.x[6:])
    predicted = base[None] @ fk @ tool[None]
    position_error = np.linalg.norm(predicted[:, :3, 3] - observed[:, :3, 3], axis=1)
    rotation_error = np.rad2deg(
        Rotation.from_matrix(np.swapaxes(predicted[:, :3, :3], 1, 2) @ observed[:, :3, :3]).magnitude()
    )
    position_quantiles = np.quantile(position_error, [0, 0.5, 0.9, 0.99, 1])
    rotation_quantiles = np.quantile(rotation_error, [0, 0.5, 0.9, 0.99, 1])
    passed = bool(
        fit.success
        and position_quantiles[2] <= position_p90_limit_m
        and rotation_quantiles[2] <= rotation_p90_limit_deg
    )
    return {
        "status": "pass" if passed else "fail",
        "manifest": str(manifest),
        "episodes": len(set(episode_ids)),
        "samples": len(fk),
        "joint_order": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "joint_unit": "radian",
        "tcp_quaternion_order": "wxyz",
        "fit_success": bool(fit.success),
        "base_transform": base.tolist(),
        "tool_transform": tool.tolist(),
        "position_error_m_q0_q50_q90_q99_q100": position_quantiles.tolist(),
        "rotation_error_deg_q0_q50_q90_q99_q100": rotation_quantiles.tolist(),
        "limits": {
            "position_p90_m": position_p90_limit_m,
            "rotation_p90_deg": rotation_p90_limit_deg,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--episode-limit", type=int, default=16)
    parser.add_argument("--samples-per-episode", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_fk_gate(
        args.manifest,
        episode_limit=args.episode_limit,
        samples_per_episode=args.samples_per_episode,
    )
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
