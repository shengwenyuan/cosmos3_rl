# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""F1 language and episode-length filtering for the DROID Fx4 manifest."""

from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml


@dataclass(frozen=True)
class CompiledTaxonomy:
    version: str
    separator: str
    vote_threshold: int
    min_length: int
    max_length: int
    family_precedence: tuple[str, ...]
    positive_families: dict[str, tuple[re.Pattern[str], ...]]
    hard_exclusions: dict[str, tuple[re.Pattern[str], ...]]


def _compile_patterns(values: Any, *, label: str) -> tuple[re.Pattern[str], ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty regex list")
    return tuple(re.compile(str(value), re.IGNORECASE) for value in values)


def load_taxonomy(path: Path) -> CompiledTaxonomy:
    with path.expanduser().resolve().open() as handle:
        raw = yaml.safe_load(handle)
    families = {
        str(name): _compile_patterns(patterns, label=f"positive_families.{name}")
        for name, patterns in raw["positive_families"].items()
    }
    precedence = tuple(str(value) for value in raw["family_precedence"])
    if set(precedence) != set(families):
        raise ValueError("family_precedence must contain every positive family exactly once")
    exclusions = {
        str(name): _compile_patterns(patterns, label=f"hard_exclusions.{name}")
        for name, patterns in raw["hard_exclusions"].items()
    }
    lengths = raw["episode_length_frames"]
    return CompiledTaxonomy(
        version=str(raw["version"]),
        separator=str(raw.get("annotation_separator", "|")),
        vote_threshold=int(raw["vote_threshold"]),
        min_length=int(lengths["min"]),
        max_length=int(lengths["max"]),
        family_precedence=precedence,
        positive_families=families,
        hard_exclusions=exclusions,
    )


def normalize_annotation(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_task_annotations(value: Any, separator: str = "|") -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        raise ValueError(f"DROID tasks must be a string or list of strings, got {type(value).__name__}")
    annotations: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError("DROID tasks list must contain only strings")
        annotations.extend(normalize_annotation(part) for part in item.split(separator) if part.strip())
    return annotations


def _matches(patterns: tuple[re.Pattern[str], ...], annotation: str) -> bool:
    return any(pattern.search(annotation) for pattern in patterns)


def classify_episode(
    *,
    length: int,
    f0_window_count: int,
    tasks_value: Any,
    taxonomy: CompiledTaxonomy,
) -> dict[str, Any]:
    annotations = parse_task_annotations(tasks_value, taxonomy.separator)
    family_votes = {family: 0 for family in taxonomy.family_precedence}
    annotation_families: list[list[str]] = []
    exclusion_hits: dict[str, list[int]] = collections.defaultdict(list)

    for index, annotation in enumerate(annotations):
        matched_families = [
            family for family in taxonomy.family_precedence if _matches(taxonomy.positive_families[family], annotation)
        ]
        annotation_families.append(matched_families)
        for family in matched_families:
            family_votes[family] += 1
        for category, patterns in taxonomy.hard_exclusions.items():
            if _matches(patterns, annotation):
                exclusion_hits[category].append(index)

    reason_codes: list[str] = []
    if f0_window_count <= 0:
        reason_codes.append("f0_not_trainable")
    if length < taxonomy.min_length:
        reason_codes.append("length_below_min")
    if length > taxonomy.max_length:
        reason_codes.append("length_above_max")
    reason_codes.extend(f"hard_exclude:{category}" for category in sorted(exclusion_hits))
    if len(annotations) < taxonomy.vote_threshold:
        reason_codes.append("insufficient_annotations")

    eligible_families = [
        family for family in taxonomy.family_precedence if family_votes[family] >= taxonomy.vote_threshold
    ]
    task_family = eligible_families[0] if eligible_families else None
    if task_family is None:
        reason_codes.append("no_two_vote_family")

    complex_votes = 0
    for matched_families in annotation_families:
        families = set(matched_families)
        if "stack_arrange" in families:
            continue
        if families == {"open_close", "push_pull_slide"}:
            continue
        if len(families) >= 2:
            complex_votes += 1
    if complex_votes >= taxonomy.vote_threshold and task_family != "stack_arrange":
        reason_codes.append("complex_multistage")

    accepted = not reason_codes
    return {
        "f1_status": "accepted" if accepted else "rejected",
        "task_family": task_family if accepted else None,
        "tasks_annotations": annotations,
        "annotation_families": annotation_families,
        "family_votes": family_votes,
        "hard_exclusion_hits": dict(exclusion_hits),
        "complex_annotation_votes": complex_votes,
        "f1_reason_codes": reason_codes if reason_codes else [f"accepted:{task_family}"],
    }


def load_episode_tasks(success_root: Path) -> dict[int, Any]:
    paths = sorted((success_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tasks: dict[int, Any] = {}
    for path in paths:
        for row in pq.read_table(path, columns=["episode_index", "tasks"]).to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index in tasks:
                raise ValueError(f"Duplicate episode_index in metadata: {episode_index}")
            tasks[episode_index] = row["tasks"]
    return tasks


def load_f0_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                record = json.loads(line)
                if int(record["episode_index"]) != len(records):
                    raise ValueError(f"F0 records must be contiguous and ordered; mismatch at line {line_number}")
                records.append(record)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_f1(
    *,
    success_root: Path,
    f0_dir: Path,
    taxonomy_path: Path,
    limit_episodes: int | None = None,
    seed: int = 42,
    progress_every: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    f0_dir = f0_dir.expanduser().resolve()
    taxonomy_path = taxonomy_path.expanduser().resolve()
    f0_summary_path = f0_dir / "f0_summary.json"
    f0_records_path = f0_dir / "f0_episodes.jsonl"
    taxonomy = load_taxonomy(taxonomy_path)
    f0_records = load_f0_records(f0_records_path)
    tasks_by_episode = load_episode_tasks(success_root)

    if limit_episodes is not None:
        if limit_episodes <= 0:
            raise ValueError("limit_episodes must be positive")
        rng = random.Random(seed)
        f0_records = sorted(
            rng.sample(f0_records, min(limit_episodes, len(f0_records))), key=lambda x: x["episode_index"]
        )

    records: list[dict[str, Any]] = []
    status_counts: collections.Counter[str] = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    reason_counts: collections.Counter[str] = collections.Counter()
    accepted_windows = 0
    examples: dict[str, list[int]] = collections.defaultdict(list)

    for record_index, f0_record in enumerate(f0_records, start=1):
        episode_index = int(f0_record["episode_index"])
        if episode_index not in tasks_by_episode:
            raise ValueError(f"Missing tasks metadata for episode {episode_index}")
        decision = classify_episode(
            length=int(f0_record["length"]),
            f0_window_count=int(f0_record["window_count"]),
            tasks_value=tasks_by_episode[episode_index],
            taxonomy=taxonomy,
        )
        record = {
            **f0_record,
            "tasks_raw": tasks_by_episode[episode_index],
            **decision,
            "taxonomy_version": taxonomy.version,
        }
        records.append(record)
        status_counts[decision["f1_status"]] += 1
        if decision["task_family"] is not None:
            family_counts[decision["task_family"]] += 1
            accepted_windows += int(f0_record["window_count"])
        for reason in decision["f1_reason_codes"]:
            reason_counts[reason] += 1
            if len(examples[reason]) < 5:
                examples[reason].append(episode_index)
        if progress_every and record_index % progress_every == 0:
            print(f"F1 progress: {record_index}/{len(f0_records)} episodes", file=sys.stderr, flush=True)

    with f0_summary_path.open() as handle:
        f0_summary = json.load(handle)
    input_files = {
        "f0_summary": {"path": str(f0_summary_path), "sha256": _sha256_file(f0_summary_path)},
        "f0_episodes": {"path": str(f0_records_path), "sha256": _sha256_file(f0_records_path)},
        "taxonomy": {"path": str(taxonomy_path), "sha256": _sha256_file(taxonomy_path)},
    }
    summary = {
        "stage": "f1",
        "taxonomy_version": taxonomy.version,
        "seed": seed,
        "limit_episodes": limit_episodes,
        "source_f0_input_fingerprint": f0_summary["input_fingerprint"],
        "counts": {
            "processed_episodes": len(records),
            "status": dict(sorted(status_counts.items())),
            "accepted_by_family": dict(sorted(family_counts.items())),
            "accepted_windows": accepted_windows,
            "reason_codes": dict(sorted(reason_counts.items())),
        },
        "examples": dict(sorted(examples.items())),
        "inputs": input_files,
    }
    summary["input_fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "source_f0_input_fingerprint": summary["source_f0_input_fingerprint"],
                "taxonomy_version": taxonomy.version,
                "seed": seed,
                "limit_episodes": limit_episodes,
                "inputs": input_files,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return summary, records
