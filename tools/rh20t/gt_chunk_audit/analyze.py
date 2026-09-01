#!/usr/bin/env python3
"""Expand captured current/raw/post/GT chunks into per-axis, per-horizon diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

AXES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper_close_fraction",
)
DEFAULT_HORIZONS = (1, 8, 16, 32)
ACTIVE_DELTA_EPS = 1e-4


def _validate_arrays(
    current: np.ndarray,
    raw: np.ndarray,
    post: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    post = np.asarray(post, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if current.ndim != 2 or current.shape[1] != len(AXES):
        raise ValueError(f"current must have shape [N,{len(AXES)}], got {current.shape}")
    expected = (len(current), raw.shape[1], len(AXES)) if raw.ndim == 3 else None
    if expected is None or raw.shape != expected or post.shape != expected or gt.shape != expected:
        raise ValueError(f"chunk shape mismatch: raw={raw.shape}, post={post.shape}, gt={gt.shape}")
    if not all(np.isfinite(value).all() for value in (current, raw, post, gt)):
        raise ValueError("current/raw/post/GT must be finite")
    return current, raw, post, gt


def _agreement(predicted: np.ndarray, target: np.ndarray) -> float | None:
    active = np.abs(target) > ACTIVE_DELTA_EPS
    if not np.any(active):
        return None
    return float(np.mean(np.sign(predicted[active]) == np.sign(target[active])))


def _scale(predicted: np.ndarray, target: np.ndarray) -> float | None:
    denominator = float(np.sum(np.abs(target)))
    if denominator <= ACTIVE_DELTA_EPS:
        return None
    return float(np.sum(np.abs(predicted)) / denominator)


def summarize_chunks(
    current: np.ndarray,
    raw: np.ndarray,
    post: np.ndarray,
    gt: np.ndarray,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    current, raw, post, gt = _validate_arrays(current, raw, post, gt)
    raw_error = raw - gt
    post_error = post - gt
    raw_delta = raw - current[:, None]
    post_delta = post - current[:, None]
    gt_delta = gt - current[:, None]
    selected = tuple(horizon for horizon in horizons if 1 <= horizon <= raw.shape[1])

    axes: dict[str, Any] = {}
    for axis_index, axis in enumerate(AXES):
        axis_raw_delta = raw_delta[..., axis_index]
        axis_post_delta = post_delta[..., axis_index]
        axis_gt_delta = gt_delta[..., axis_index]
        axes[axis] = {
            "raw_mae": float(np.mean(np.abs(raw_error[..., axis_index]))),
            "post_mae": float(np.mean(np.abs(post_error[..., axis_index]))),
            "raw_direction_agreement": _agreement(axis_raw_delta, axis_gt_delta),
            "post_direction_agreement": _agreement(axis_post_delta, axis_gt_delta),
            "raw_motion_scale": _scale(axis_raw_delta, axis_gt_delta),
            "post_motion_scale": _scale(axis_post_delta, axis_gt_delta),
            "horizons": {
                str(horizon): {
                    "raw_mae": float(np.mean(np.abs(raw_error[:, horizon - 1, axis_index]))),
                    "post_mae": float(np.mean(np.abs(post_error[:, horizon - 1, axis_index]))),
                }
                for horizon in selected
            },
        }
    return {"queries": len(current), "chunk_size": raw.shape[1], "axes": axes}


def _read_capture(samples_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(samples_path) as capture:
        required = {"current", "raw_prediction", "prediction", "target"}
        missing = required.difference(capture.files)
        if missing:
            raise ValueError(f"capture is missing arrays: {sorted(missing)}")
        return _validate_arrays(
            capture["current"],
            capture["raw_prediction"],
            capture["prediction"],
            capture["target"],
        )


def _read_indices(report_path: Path | None, count: int) -> tuple[list[int], dict[str, Any]]:
    if report_path is None:
        return list(range(count)), {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    indices = [int(index) for index in report.get("indices", [])]
    if len(indices) != count:
        raise ValueError(f"report has {len(indices)} indices for {count} captured queries")
    return indices, report


def _write_query(
    output: Path,
    *,
    ordinal: int,
    dataset_index: int,
    current: np.ndarray,
    raw: np.ndarray,
    post: np.ndarray,
    gt: np.ndarray,
) -> None:
    query_dir = output / f"query_{ordinal:03d}_dataset_{dataset_index}"
    query_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(query_dir / "chunks.npz", current=current, raw=raw, post=post, gt=gt)
    metrics = summarize_chunks(current[None], raw[None], post[None], gt[None])
    metrics.update({"ordinal": ordinal, "dataset_index": dataset_index})
    (query_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _write_rows(
    output: Path,
    indices: list[int],
    current: np.ndarray,
    raw: np.ndarray,
    post: np.ndarray,
    gt: np.ndarray,
) -> None:
    fields = (
        "ordinal",
        "dataset_index",
        "horizon",
        "axis",
        "current",
        "raw",
        "post",
        "gt",
        "raw_error",
        "post_error",
        "raw_delta",
        "post_delta",
        "gt_delta",
    )
    with (output / "per_axis_horizon.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ordinal, dataset_index in enumerate(indices):
            for horizon in range(raw.shape[1]):
                for axis_index, axis in enumerate(AXES):
                    current_value = float(current[ordinal, axis_index])
                    raw_value = float(raw[ordinal, horizon, axis_index])
                    post_value = float(post[ordinal, horizon, axis_index])
                    gt_value = float(gt[ordinal, horizon, axis_index])
                    writer.writerow(
                        {
                            "ordinal": ordinal,
                            "dataset_index": dataset_index,
                            "horizon": horizon + 1,
                            "axis": axis,
                            "current": current_value,
                            "raw": raw_value,
                            "post": post_value,
                            "gt": gt_value,
                            "raw_error": raw_value - gt_value,
                            "post_error": post_value - gt_value,
                            "raw_delta": raw_value - current_value,
                            "post_delta": post_value - current_value,
                            "gt_delta": gt_value - current_value,
                        }
                    )


def materialize_audit(samples_path: Path, report_path: Path | None, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output}")
    current, raw, post, gt = _read_capture(samples_path)
    indices, report = _read_indices(report_path, len(current))
    output.mkdir(parents=True)
    for ordinal, dataset_index in enumerate(indices):
        _write_query(
            output,
            ordinal=ordinal,
            dataset_index=dataset_index,
            current=current[ordinal],
            raw=raw[ordinal],
            post=post[ordinal],
            gt=gt[ordinal],
        )
    _write_rows(output, indices, current, raw, post, gt)
    summary = summarize_chunks(current, raw, post, gt)
    summary.update(
        {
            "samples": str(samples_path),
            "capture_report": str(report_path) if report_path else None,
            "checkpoint": report.get("checkpoint"),
            "dataset_indices": indices,
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = materialize_audit(args.samples, args.report, args.output)
    print(json.dumps(summary, indent=2))
    print(f"audit={args.output}")


if __name__ == "__main__":
    main()
