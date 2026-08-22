#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail closed before launching DROID Fx4 Stage-1 EEF training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import tomllib


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _validate_checksum_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing checksum file: {path}")
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        target = path.parent / filename.strip()
        if not target.is_file() or _sha256(target) != digest:
            raise ValueError(f"checksum mismatch: {target}")


def _all_gates_pass(gates: dict[str, Any]) -> bool:
    return all(value if isinstance(value, bool) else value == 0 for value in gates.values())


def validate_topology(toml_path: Path, world_size: int) -> dict[str, int]:
    with toml_path.open("rb") as handle:
        config = tomllib.load(handle)
    if config["job"]["experiment"] != "action_policy_droid_eef_edge":
        raise ValueError("TOML does not select action_policy_droid_eef_edge")
    parallelism = config["model"]["parallelism"]
    configured_world_size = math.prod(
        int(parallelism[key])
        for key in (
            "data_parallel_shard_degree",
            "data_parallel_replicate_degree",
            "context_parallel_shard_degree",
            "cfg_parallel_shard_degree",
        )
    )
    if configured_world_size != world_size:
        raise ValueError(f"TOML topology has {configured_world_size} ranks, launch requested {world_size}")
    per_rank_batch = int(config["dataloader_train"]["max_samples_per_batch"])
    grad_accum = int(config["trainer"]["grad_accum_iter"])
    global_batch = per_rank_batch * world_size * grad_accum
    expected_global_batch = {4: 32, 8: 64, 16: 128, 32: 256, 64: 512}[world_size]
    if global_batch != expected_global_batch:
        raise ValueError(f"Stage-1 topology requires global batch {expected_global_batch}, got {global_batch}")
    return {
        "world_size": world_size,
        "per_rank_batch": per_rank_batch,
        "grad_accum": grad_accum,
        "global_batch": global_batch,
    }


def validate_artifacts(dataset_root: Path, manifest_dir: Path, action_stats_path: Path) -> dict[str, Any]:
    success_root = dataset_root / "success" if (dataset_root / "success").is_dir() else dataset_root
    required_dataset_paths = (
        success_root / "meta/info.json",
        success_root / "meta/episodes",
        success_root / "data",
        success_root / "videos/observation.image.wrist_image_left",
        success_root / "videos/observation.image.exterior_image_1_left",
        success_root / "videos/observation.image.exterior_image_2_left",
    )
    missing = [str(path) for path in required_dataset_paths if not path.exists()]
    if missing:
        raise ValueError(f"DROID dataset is incomplete: {missing}")

    _validate_checksum_file(manifest_dir / "F3_SHA256SUMS")
    _validate_checksum_file(manifest_dir / "SHA256SUMS")
    _validate_checksum_file(action_stats_path.with_name("ACTION_STATS_SHA256SUMS"))
    f3 = _load_json(manifest_dir / "f3_summary.json")
    if not _all_gates_pass(f3["gates"]):
        raise ValueError(f"F3 gates did not pass: {f3['gates']}")
    final_summary = _load_json(manifest_dir / "summary.json")
    if final_summary.get("status") != "pass":
        raise ValueError(f"final Fx4 gate is not pass: {final_summary.get('status')!r}")

    train_manifest = manifest_dir / "manifest_train.jsonl"
    stats = _load_json(action_stats_path)
    metadata = stats["metadata"]
    expected_metadata = {
        "action_dim": 10,
        "chunk_length": 32,
        "normalization": "quantile_rot",
        "apply_forward_clamp": False,
    }
    mismatches = {
        key: (metadata.get(key), value) for key, value in expected_metadata.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"action statistics contract mismatch: {mismatches}")
    if metadata.get("manifest_sha256") != _sha256(train_manifest):
        raise ValueError("action statistics were not computed from the selected train manifest")
    for stats_key in ("global", "global_raw"):
        for field in ("mean", "std", "min", "max", "q01", "q99"):
            values = stats[stats_key][field]
            if len(values) != 10 or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"invalid {stats_key}.{field} action statistics")

    return {
        "dataset_root": str(success_root),
        "manifest": str(train_manifest),
        "manifest_sha256": _sha256(train_manifest),
        "action_stats": str(action_stats_path),
        "action_stats_sha256": _sha256(action_stats_path),
        "pipeline_status": final_summary["status"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(4, 8, 16, 32, 64), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = {
        "topology": validate_topology(args.toml.expanduser().resolve(), args.world_size),
        "artifacts": validate_artifacts(
            args.dataset_root.expanduser().resolve(),
            args.manifest_dir.expanduser().resolve(),
            args.action_stats.expanduser().resolve(),
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
