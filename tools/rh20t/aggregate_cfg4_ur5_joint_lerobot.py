# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Aggregate ordered RH20T LeRobot shards into one local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from tools.rh20t.rh20t_io import atomic_write_json


def discover_shards(root: Path, expected_shards: int) -> list[Path]:
    shards = sorted(path for path in root.glob("shard-*.staging") if path.is_dir())
    expected_names = [f"shard-{index:02d}.staging" for index in range(expected_shards)]
    if [path.name for path in shards] != expected_names:
        raise ValueError(f"Shard set mismatch: found={[path.name for path in shards]}, expected={expected_names}")
    return shards


def aggregate(*, shards_root: Path, output: Path, repo_id: str, expected_shards: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Aggregate output already exists: {output}")
    shards = discover_shards(shards_root, expected_shards)
    metadata = [
        LeRobotDatasetMetadata(repo_id=f"local/shard-{index:02d}", root=root, revision="local")
        for index, root in enumerate(shards)
    ]
    for index, meta in enumerate(metadata):
        if int(meta.total_episodes) <= 0:
            raise ValueError(f"Shard {index} contains no episodes")
    started = time.time()
    aggregate_datasets(
        repo_ids=[f"local/shard-{index:02d}" for index in range(len(shards))],
        roots=shards,
        aggr_repo_id=repo_id,
        aggr_root=output,
    )
    result = LeRobotDatasetMetadata(repo_id=repo_id, root=output, revision="local")
    report = {
        "status": "complete",
        "shards_root": str(shards_root),
        "output": str(output),
        "repo_id": repo_id,
        "shards": [
            {"root": str(root), "episodes": int(meta.total_episodes), "frames": int(meta.total_frames)}
            for root, meta in zip(shards, metadata, strict=True)
        ],
        "episodes": int(result.total_episodes),
        "frames": int(result.total_frames),
        "elapsed_s": round(time.time() - started, 3),
    }
    if report["episodes"] != sum(row["episodes"] for row in report["shards"]):
        raise ValueError("Aggregated episode count does not equal shard sum")
    if report["frames"] != sum(row["frames"] for row in report["shards"]):
        raise ValueError("Aggregated frame count does not equal shard sum")
    atomic_write_json(output / "conversion" / "aggregate_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = aggregate(
        shards_root=args.shards_root,
        output=args.output,
        repo_id=args.repo_id,
        expected_shards=args.expected_shards,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
