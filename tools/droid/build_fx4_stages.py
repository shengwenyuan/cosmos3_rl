#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build immutable F2-F4 stages for the DROID Fx4 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.droid.build_fx4_manifest import resolve_success_root
from tools.droid.fx4_f2 import F2Config, build_f2_motion, build_f2_video, write_f2_motion, write_f2_video
from tools.droid.fx4_f3 import F3Config, build_f3, write_f3
from tools.droid.fx4_f4 import build_f4, read_review_gate
from tools.droid.fx4_io import refuse_source_output, validate_checksums

STAGE_FILES = {
    "f2-motion": ("f2_motion_summary.json", "F2_MOTION_SHA256SUMS"),
    "f2-video": ("f2_summary.json", "F2_SHA256SUMS"),
    "f3": ("f3_summary.json", "F3_SHA256SUMS"),
    "f4": ("f4_summary.json", "F4_SHA256SUMS"),
}


def _existing_stage(output_dir: Path, stage: str, resume: bool) -> dict | None:
    summary_name, checksums_name = STAGE_FILES[stage]
    summary_path = output_dir / summary_name
    checksums_path = output_dir / checksums_name
    any_exists = summary_path.exists() or checksums_path.exists()
    if not any_exists:
        return None
    if not resume:
        raise FileExistsError(f"{stage} output exists; pass --resume or use a new output directory")
    if not summary_path.is_file() or not checksums_path.is_file():
        raise ValueError(f"Incomplete {stage} output cannot be resumed")
    validate_checksums(checksums_path)
    with summary_path.open() as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGE_FILES, "f4-gate"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--taxonomy-path", type=Path, default=Path(__file__).with_name("fx4_f1_taxonomy.yaml"))
    parser.add_argument("--qc-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    success_root = resolve_success_root(args.dataset_root)
    output_dir = args.manifest_dir.expanduser().resolve()
    refuse_source_output(success_root, output_dir)
    if args.dry_run and args.resume:
        raise SystemExit("--dry-run cannot be combined with --resume")
    if not args.dry_run and args.limit_episodes is not None:
        raise SystemExit("--limit-episodes is only allowed with --dry-run")
    if args.stage == "f4-gate":
        print(json.dumps(read_review_gate(output_dir / "qc_review.csv"), indent=2, sort_keys=True))
        return 0

    if not args.dry_run:
        existing = _existing_stage(output_dir, args.stage, args.resume)
        if existing is not None:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0

    if args.stage == "f2-motion":
        summary, records, thresholds = build_f2_motion(
            success_root=success_root,
            f1_dir=output_dir,
            output_dir=output_dir,
            config=F2Config(),
            limit_episodes=args.limit_episodes,
        )
        if not args.dry_run:
            write_f2_motion(output_dir, summary, records, thresholds)
    elif args.stage == "f2-video":
        summary, records = build_f2_video(
            success_root=success_root,
            f2_dir=output_dir,
            config=F2Config(),
            limit_episodes=args.limit_episodes,
            checkpoint_dir=None if args.dry_run else output_dir / ".f2_video_checkpoints",
            resume=args.resume,
            workers=args.workers,
        )
        if not args.dry_run:
            write_f2_video(output_dir, summary, records)
    elif args.stage == "f3":
        summary, train, val = build_f3(
            success_root=success_root,
            f2_dir=output_dir,
            config=F3Config(seed=args.seed),
        )
        if not args.dry_run:
            write_f3(output_dir, summary, train, val)
    else:
        if args.dry_run:
            raise SystemExit("F4 renders review artifacts and does not support --dry-run")
        qc_root = args.qc_root or output_dir.parent.parent / "qc" / output_dir.name
        summary = build_f4(
            success_root=success_root,
            manifest_dir=output_dir,
            taxonomy_path=args.taxonomy_path,
            qc_root=qc_root,
            seed=args.seed,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
