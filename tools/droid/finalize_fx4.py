#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Assemble the final Fx4 audit summary and checksum set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tools.droid.fx4_f4 import read_review_gate
from tools.droid.fx4_io import atomic_write_text, sha256_file, write_checksums, write_json


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def build_final_summary(manifest_dir: Path, action_stats_path: Path) -> dict[str, Any]:
    summaries = {
        stage: _load(manifest_dir / filename)
        for stage, filename in (
            ("f0", "f0_summary.json"),
            ("f1", "f1_summary.json"),
            ("f2_motion", "f2_motion_summary.json"),
            ("f2", "f2_summary.json"),
            ("f3", "f3_summary.json"),
            ("f4", "f4_summary.json"),
        )
    }
    review_path = manifest_dir / "qc_review.csv"
    f4_review = read_review_gate(review_path) if review_path.is_file() else summaries["f4"]["gates"]["semantic_review"]
    f3_gates_pass = all(value if isinstance(value, bool) else value == 0 for value in summaries["f3"]["gates"].values())
    gates = {
        "f0_official_counts": bool(summaries["f0"].get("expected_counts_match")),
        "f2_motion": all(summaries["f2_motion"]["gates"].values()),
        "f2_video_decode": bool(summaries["f2"]["gates"]["video_decode_failure_lt_0_001"]),
        "f3_split_and_timeline": f3_gates_pass,
        "f4_coverage": bool(summaries["f4"]["gates"]["coverage_pass"]),
        "f4_semantic_review": f4_review["status"],
    }
    hard_gate_values = [value for key, value in gates.items() if key != "f4_semantic_review"]
    if not all(hard_gate_values) or f4_review["status"] == "fail":
        status = "fail"
    elif f4_review["status"] == "pending":
        status = "pending_human_review"
    else:
        status = "pass"
    return {
        "pipeline": manifest_dir.name,
        "status": status,
        "gates": gates,
        "f4_semantic_review": f4_review,
        "stage_counts": {stage: summary.get("counts", {}) for stage, summary in summaries.items()},
        "action_stats": {"path": str(action_stats_path), "sha256": sha256_file(action_stats_path)},
    }


def finalize(manifest_dir: Path, action_stats_path: Path) -> dict[str, Any]:
    summary_path = manifest_dir / "summary.json"
    thresholds_path = manifest_dir / "thresholds.json"
    if summary_path.exists() or thresholds_path.exists() or (manifest_dir / "SHA256SUMS").exists():
        raise FileExistsError("Final Fx4 output already exists; use a new version instead of overwriting")
    atomic_write_text(thresholds_path, (manifest_dir / "f2_thresholds.json").read_text())
    summary = build_final_summary(manifest_dir, action_stats_path)
    write_json(summary_path, summary)
    files = (
        manifest_dir / "manifest_train.jsonl",
        manifest_dir / "manifest_val.jsonl",
        summary_path,
        manifest_dir / "taxonomy.yaml",
        thresholds_path,
        manifest_dir / "qc_review.csv",
        manifest_dir / "rejected_summary.json",
    )
    write_checksums(manifest_dir / "SHA256SUMS", files)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize(args.manifest_dir.expanduser().resolve(), args.action_stats.expanduser().resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
