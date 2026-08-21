#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the metadata-only F0 layer of the DROID Franka Fx4 manifest.

F0 reproduces the official ``keep_ranges_1_0_1.json`` indexing semantics
without opening data parquet payloads or videos.  ``--dry-run`` is strictly
read-only and prints the summary to stdout.  A non-dry run writes an immutable
F0 episode ledger and its input fingerprints to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from tools.droid.fx4_f1 import build_f1

DROID_GCS_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full/"
DEFAULT_CHUNK_LENGTH = 32
DEFAULT_EXPECTED_MATCHED_EPISODES = 56_804
DEFAULT_EXPECTED_FILTER_RANGES = 75_558
DEFAULT_EXPECTED_VALID_EPISODES = 56_687
DEFAULT_EXPECTED_VALID_RANGES = 68_494
DEFAULT_EXPECTED_WINDOWS = 12_538_157


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_index: int
    episode_id: str
    length: int
    dataset_from_index: int
    dataset_to_index: int


@dataclass(frozen=True)
class RangeResult:
    source_start: int
    source_end: int
    clipped_start: int
    clipped_end: int
    window_count: int
    legacy_inclusive_window_count: int
    rejection_reason: str | None


def canonical_filter_key(episode_id: str) -> str:
    """Return the exact DROID key used by the current training adapter."""
    base = f"{DROID_GCS_PREFIX}{episode_id}"
    return f"{base}/recordings/MP4--{base}/trajectory.h5"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_success_root(dataset_root: Path) -> Path:
    """Accept either the repository root or its ``success`` split root."""
    dataset_root = dataset_root.expanduser().resolve()
    candidates = (dataset_root, dataset_root / "success")
    for candidate in candidates:
        if (candidate / "meta" / "info.json").is_file():
            return candidate
    raise FileNotFoundError(f"No LeRobot info.json under {dataset_root} or {dataset_root / 'success'}")


def load_episode_metadata(success_root: Path) -> tuple[list[EpisodeMetadata], list[Path]]:
    metadata_paths = sorted((success_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {success_root}")

    columns = ["episode_index", "episode_id", "length", "dataset_from_index", "dataset_to_index"]
    episodes: list[EpisodeMetadata] = []
    for path in metadata_paths:
        for row in pq.read_table(path, columns=columns).to_pylist():
            episode = EpisodeMetadata(
                episode_index=int(row["episode_index"]),
                episode_id=str(row["episode_id"]),
                length=int(row["length"]),
                dataset_from_index=int(row["dataset_from_index"]),
                dataset_to_index=int(row["dataset_to_index"]),
            )
            if episode.length != episode.dataset_to_index - episode.dataset_from_index:
                raise ValueError(
                    f"Episode {episode.episode_index} length mismatch: "
                    f"length={episode.length}, span={episode.dataset_to_index - episode.dataset_from_index}"
                )
            episodes.append(episode)

    episodes.sort(key=lambda item: item.episode_index)
    expected_indices = list(range(len(episodes)))
    actual_indices = [item.episode_index for item in episodes]
    if actual_indices != expected_indices:
        raise ValueError("episode_index must be unique, contiguous, and zero-based")
    return episodes, metadata_paths


def evaluate_range(start: int, end: int, episode_length: int, chunk_length: int) -> RangeResult:
    """Apply the current DROID loader's half-open range/window semantics."""
    if end < start:
        raise ValueError(f"Invalid keep range [{start}, {end}): end precedes start")
    episode_valid_len = max(0, episode_length - chunk_length)
    clipped_start = max(start, 0)
    clipped_end = min(end - chunk_length, episode_valid_len)
    window_count = max(0, clipped_end - clipped_start)
    # Historical probes used an inclusive-start formula here.  It requires
    # only 32 source frames for a 32-action sample, while the current loader
    # fetches 33 observations, so this is diagnostic only and never trainable.
    legacy_inclusive_window_count = max(0, clipped_end - clipped_start + 1)

    rejection_reason = None
    if window_count == 0:
        if episode_length <= chunk_length:
            rejection_reason = "episode_too_short"
        elif end - start <= chunk_length:
            rejection_reason = "range_too_short"
        else:
            rejection_reason = "range_outside_episode"

    return RangeResult(
        source_start=start,
        source_end=end,
        clipped_start=clipped_start,
        clipped_end=clipped_end,
        window_count=window_count,
        legacy_inclusive_window_count=legacy_inclusive_window_count,
        rejection_reason=rejection_reason,
    )


def _parse_ranges(value: Any, *, filter_key: str) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        raise ValueError(f"Filter entry {filter_key!r} must be a list")
    parsed: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Filter entry {filter_key!r} range {index} must be [start, end]")
        start, end = item
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"Filter entry {filter_key!r} range {index} must use integer bounds")
        parsed.append((start, end))
    return parsed


def discover_dataset_revision(dataset_root: Path) -> str | None:
    metadata_root = dataset_root / ".cache" / "huggingface" / "download"
    revisions: set[str] = set()
    for path in metadata_root.rglob("*.metadata") if metadata_root.is_dir() else ():
        with path.open(errors="replace") as handle:
            revision = handle.readline().strip()
        if revision:
            revisions.add(revision)
    if len(revisions) > 1:
        raise ValueError(f"Dataset cache contains multiple revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def build_f0(
    *,
    dataset_root: Path,
    filter_path: Path,
    chunk_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if chunk_length <= 0:
        raise ValueError(f"chunk_length must be positive, got {chunk_length}")
    success_root = resolve_success_root(dataset_root)
    dataset_root = success_root.parent if success_root.name == "success" else success_root
    filter_path = filter_path.expanduser().resolve()
    if not filter_path.is_file():
        raise FileNotFoundError(filter_path)

    with filter_path.open() as handle:
        filter_dict = json.load(handle)
    if not isinstance(filter_dict, dict):
        raise ValueError("keep-ranges JSON must be an object keyed by DROID episode URI")

    episodes, metadata_paths = load_episode_metadata(success_root)
    info_path = success_root / "meta" / "info.json"
    with info_path.open() as handle:
        info = json.load(handle)
    if int(info["total_episodes"]) != len(episodes):
        raise ValueError(f"info.json declares {info['total_episodes']} episodes, metadata has {len(episodes)}")

    records: list[dict[str, Any]] = []
    matched_episodes = 0
    valid_episodes = 0
    filter_ranges = 0
    valid_ranges = 0
    windows = 0
    legacy_inclusive_valid_episodes = 0
    legacy_inclusive_valid_ranges = 0
    legacy_inclusive_windows = 0

    for episode in episodes:
        filter_key = canonical_filter_key(episode.episode_id)
        raw_ranges = filter_dict.get(filter_key)
        ranges: list[RangeResult] = []
        if raw_ranges is not None:
            matched_episodes += 1
            parsed_ranges = _parse_ranges(raw_ranges, filter_key=filter_key)
            filter_ranges += len(parsed_ranges)
            ranges = [evaluate_range(start, end, episode.length, chunk_length) for start, end in parsed_ranges]

        episode_windows = sum(item.window_count for item in ranges)
        episode_valid_ranges = sum(item.window_count > 0 for item in ranges)
        episode_legacy_windows = sum(item.legacy_inclusive_window_count for item in ranges)
        episode_legacy_valid_ranges = sum(item.legacy_inclusive_window_count > 0 for item in ranges)
        if episode_windows > 0:
            valid_episodes += 1
        if episode_legacy_windows > 0:
            legacy_inclusive_valid_episodes += 1
        valid_ranges += episode_valid_ranges
        windows += episode_windows
        legacy_inclusive_valid_ranges += episode_legacy_valid_ranges
        legacy_inclusive_windows += episode_legacy_windows
        records.append(
            {
                "episode_index": episode.episode_index,
                "episode_id": episode.episode_id,
                "length": episode.length,
                "dataset_from_index": episode.dataset_from_index,
                "dataset_to_index": episode.dataset_to_index,
                "filter_key": filter_key,
                "filter_status": "matched" if raw_ranges is not None else "missing",
                "ranges": [asdict(item) for item in ranges],
                "valid_range_count": episode_valid_ranges,
                "window_count": episode_windows,
            }
        )

    metadata_sha256 = {str(path.relative_to(dataset_root)): sha256_file(path) for path in metadata_paths}
    input_files = {
        "filter": {"path": str(filter_path), "sha256": sha256_file(filter_path)},
        "info": {"path": str(info_path), "sha256": sha256_file(info_path)},
        "episode_metadata_sha256": metadata_sha256,
    }
    summary: dict[str, Any] = {
        "stage": "f0",
        "dataset_root": str(dataset_root),
        "success_root": str(success_root),
        "dataset_revision": discover_dataset_revision(dataset_root),
        "chunk_length": chunk_length,
        "observation_frames": chunk_length + 1,
        "range_semantics": "half_open_[start,end)",
        "counts": {
            "dataset_episodes": len(episodes),
            "filter_keys": len(filter_dict),
            "matched_episodes": matched_episodes,
            "missing_episodes": len(episodes) - matched_episodes,
            "valid_episodes": valid_episodes,
            "empty_matched_episodes": matched_episodes - valid_episodes,
            "filter_ranges": filter_ranges,
            "valid_ranges": valid_ranges,
            "windows": windows,
            "legacy_inclusive_valid_episodes": legacy_inclusive_valid_episodes,
            "legacy_inclusive_valid_ranges": legacy_inclusive_valid_ranges,
            "legacy_inclusive_windows": legacy_inclusive_windows,
        },
        "inputs": input_files,
    }
    summary["input_fingerprint"] = _json_sha256(
        {
            "dataset_revision": summary["dataset_revision"],
            "chunk_length": chunk_length,
            "inputs": input_files,
        }
    )
    return summary, records


def expected_mismatches(
    summary: dict[str, Any],
    *,
    expected_matched_episodes: int,
    expected_filter_ranges: int,
    expected_valid_episodes: int,
    expected_valid_ranges: int,
    expected_windows: int,
) -> dict[str, dict[str, int]]:
    counts = summary["counts"]
    expected = {
        "matched_episodes": expected_matched_episodes,
        "filter_ranges": expected_filter_ranges,
        "valid_episodes": expected_valid_episodes,
        "valid_ranges": expected_valid_ranges,
        "windows": expected_windows,
    }
    return {
        key: {"expected": value, "actual": int(counts[key])}
        for key, value in expected.items()
        if int(counts[key]) != value
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def write_f0_outputs(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "f0_summary.json"
    records_path = output_dir / "f0_episodes.jsonl"
    checksums_path = output_dir / "F0_SHA256SUMS"

    _atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(records_path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    checksums = f"{sha256_file(summary_path)}  {summary_path.name}\n{sha256_file(records_path)}  {records_path.name}\n"
    _atomic_write_text(checksums_path, checksums)


def _validate_complete_f0_output(output_dir: Path, input_fingerprint: str) -> dict[str, Any] | None:
    summary_path = output_dir / "f0_summary.json"
    records_path = output_dir / "f0_episodes.jsonl"
    checksums_path = output_dir / "F0_SHA256SUMS"
    paths = (summary_path, records_path, checksums_path)
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        return None

    with summary_path.open() as handle:
        existing_summary = json.load(handle)
    if existing_summary.get("input_fingerprint") != input_fingerprint:
        raise ValueError("Existing F0 output has a different input fingerprint; use a new output directory")

    expected_hashes: dict[str, str] = {}
    for line in checksums_path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        expected_hashes[filename.strip()] = digest
    for path in (summary_path, records_path):
        if expected_hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"Existing F0 output checksum mismatch: {path}")
    return existing_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("f0", "f1"), default="f0")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--filter-path", type=Path)
    parser.add_argument("--f0-dir", type=Path)
    parser.add_argument("--taxonomy-path", type=Path, default=Path(__file__).with_name("fx4_f1_taxonomy.yaml"))
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-length", type=int, default=DEFAULT_CHUNK_LENGTH)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print summary and do not create or modify output files."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse a complete F0 output with the same input fingerprint."
    )
    parser.add_argument("--check-expected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-matched-episodes", type=int, default=DEFAULT_EXPECTED_MATCHED_EPISODES)
    parser.add_argument("--expected-filter-ranges", type=int, default=DEFAULT_EXPECTED_FILTER_RANGES)
    parser.add_argument("--expected-valid-episodes", type=int, default=DEFAULT_EXPECTED_VALID_EPISODES)
    parser.add_argument("--expected-valid-ranges", type=int, default=DEFAULT_EXPECTED_VALID_RANGES)
    parser.add_argument("--expected-windows", type=int, default=DEFAULT_EXPECTED_WINDOWS)
    return parser


def _run_f1(args: argparse.Namespace) -> int:
    if args.f0_dir is None:
        raise SystemExit("--f0-dir is required for --stage f1")
    if not args.dry_run:
        raise SystemExit("F1 formal output is not enabled yet; use --dry-run for development smoke")
    if args.resume:
        raise SystemExit("--resume is not supported by the F1 development smoke")

    success_root = resolve_success_root(args.dataset_root)
    summary, _ = build_f1(
        success_root=success_root,
        f0_dir=args.f0_dir,
        taxonomy_path=args.taxonomy_path,
        limit_episodes=args.limit_episodes,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "f1":
        return _run_f1(args)
    if not args.dry_run and args.output_dir is None:
        raise SystemExit("--output-dir is required unless --dry-run is set")
    if args.dry_run and args.resume:
        raise SystemExit("--resume cannot be combined with --dry-run")
    if args.filter_path is None:
        raise SystemExit("--filter-path is required for --stage f0")

    summary, records = build_f0(
        dataset_root=args.dataset_root,
        filter_path=args.filter_path,
        chunk_length=args.chunk_length,
    )
    mismatches = expected_mismatches(
        summary,
        expected_matched_episodes=args.expected_matched_episodes,
        expected_filter_ranges=args.expected_filter_ranges,
        expected_valid_episodes=args.expected_valid_episodes,
        expected_valid_ranges=args.expected_valid_ranges,
        expected_windows=args.expected_windows,
    )
    summary["expected_counts"] = {
        "matched_episodes": args.expected_matched_episodes,
        "filter_ranges": args.expected_filter_ranges,
        "valid_episodes": args.expected_valid_episodes,
        "valid_ranges": args.expected_valid_ranges,
        "windows": args.expected_windows,
    }
    summary["expected_count_mismatches"] = mismatches
    summary["expected_counts_match"] = not mismatches

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        success_root = resolve_success_root(args.dataset_root)
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir == success_root or success_root in output_dir.parents:
            raise ValueError(f"Refusing to write F0 outputs inside the read-only source dataset: {output_dir}")
        existing_names = ("f0_summary.json", "f0_episodes.jsonl", "F0_SHA256SUMS")
        has_existing_f0 = any((output_dir / name).exists() for name in existing_names)
        if has_existing_f0 and not args.resume:
            raise FileExistsError(f"F0 output already exists under {output_dir}; pass --resume or use a new directory")
        if args.resume:
            existing_summary = _validate_complete_f0_output(output_dir, summary["input_fingerprint"])
            if existing_summary is not None:
                print(json.dumps(existing_summary, indent=2, sort_keys=True))
                return 0 if not args.check_expected or existing_summary.get("expected_counts_match") else 2
        write_f0_outputs(output_dir, summary, records)
        print(json.dumps(summary, indent=2, sort_keys=True))

    if args.check_expected and mismatches:
        print(f"F0 expected-count mismatch: {json.dumps(mismatches, sort_keys=True)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
