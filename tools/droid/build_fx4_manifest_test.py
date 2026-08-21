# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.droid.build_fx4_manifest import (
    build_f0,
    canonical_filter_key,
    evaluate_range,
    main,
    write_f0_outputs,
    write_f1_outputs,
)


def _make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "Cosmos3-DROID"
    success_root = dataset_root / "success"
    episodes_root = success_root / "meta" / "episodes" / "chunk-000"
    episodes_root.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 15,
        "total_episodes": 3,
        "total_frames": 160,
    }
    (success_root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1, 2],
                "episode_id": ["lab/success/episode0", "lab/success/episode1", "lab/success/episode2"],
                "length": [100, 40, 20],
                "dataset_from_index": [0, 100, 140],
                "dataset_to_index": [100, 140, 160],
            }
        ),
        episodes_root / "file-000.parquet",
    )
    filter_path = tmp_path / "keep_ranges.json"
    filter_path.write_text(
        json.dumps(
            {
                canonical_filter_key("lab/success/episode0"): [[0, 100]],
                canonical_filter_key("lab/success/episode1"): [[0, 32], [0, 33]],
            }
        )
    )
    return dataset_root, filter_path


def test_canonical_filter_key_matches_droid_loader_contract() -> None:
    episode_id = "AUTOLab/success/2023-07-07/example"
    prefix = "gs://xembodiment_data/r2d2/r2d2-data-full/"
    assert canonical_filter_key(episode_id) == (
        f"{prefix}{episode_id}/recordings/MP4--{prefix}{episode_id}/trajectory.h5"
    )


@pytest.mark.parametrize(
    ("start", "end", "episode_length", "expected_windows", "expected_reason"),
    [
        (0, 100, 100, 68, None),
        (10, 50, 100, 8, None),
        (-5, 120, 100, 68, None),
        (0, 32, 100, 0, "range_too_short"),
        (0, 33, 100, 1, None),
        (0, 100, 20, 0, "episode_too_short"),
    ],
)
def test_evaluate_range_uses_half_open_loader_semantics(
    start: int,
    end: int,
    episode_length: int,
    expected_windows: int,
    expected_reason: str | None,
) -> None:
    result = evaluate_range(start, end, episode_length, chunk_length=32)
    assert result.window_count == expected_windows
    assert result.rejection_reason == expected_reason


def test_build_f0_counts_matches_and_keeps_rejection_reasons(tmp_path: Path) -> None:
    dataset_root, filter_path = _make_dataset(tmp_path)
    summary, records = build_f0(dataset_root=dataset_root, filter_path=filter_path, chunk_length=32)

    assert summary["counts"] == {
        "dataset_episodes": 3,
        "filter_keys": 2,
        "matched_episodes": 2,
        "missing_episodes": 1,
        "valid_episodes": 2,
        "empty_matched_episodes": 0,
        "filter_ranges": 3,
        "valid_ranges": 2,
        "windows": 69,
        "legacy_inclusive_valid_episodes": 2,
        "legacy_inclusive_valid_ranges": 3,
        "legacy_inclusive_windows": 72,
    }
    assert records[1]["ranges"][0]["rejection_reason"] == "range_too_short"
    assert records[2]["filter_status"] == "missing"
    assert records[2]["ranges"] == []


def test_dry_run_does_not_create_output_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_root, filter_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "must_not_exist"
    rc = main(
        [
            "--dataset-root",
            str(dataset_root),
            "--filter-path",
            str(filter_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--no-check-expected",
        ]
    )
    assert rc == 0
    assert not output_dir.exists()
    assert json.loads(capsys.readouterr().out)["counts"]["windows"] == 69


def test_write_outputs_records_checksums(tmp_path: Path) -> None:
    dataset_root, filter_path = _make_dataset(tmp_path)
    summary, records = build_f0(dataset_root=dataset_root, filter_path=filter_path, chunk_length=32)
    output_dir = tmp_path / "derived" / "f0"

    write_f0_outputs(output_dir, summary, records)

    assert (output_dir / "f0_summary.json").is_file()
    assert len((output_dir / "f0_episodes.jsonl").read_text().splitlines()) == 3
    checksums = (output_dir / "F0_SHA256SUMS").read_text()
    assert "f0_summary.json" in checksums
    assert "f0_episodes.jsonl" in checksums


def test_write_f1_outputs_records_checksums(tmp_path: Path) -> None:
    output_dir = tmp_path / "derived" / "f1"
    summary = {"stage": "f1", "input_fingerprint": "abc"}
    records = [{"episode_index": 0, "f1_status": "accepted"}]

    write_f1_outputs(output_dir, summary, records)

    assert json.loads((output_dir / "f1_summary.json").read_text())["stage"] == "f1"
    assert len((output_dir / "f1_episodes.jsonl").read_text().splitlines()) == 1
    checksums = (output_dir / "F1_SHA256SUMS").read_text()
    assert "f1_summary.json" in checksums
    assert "f1_episodes.jsonl" in checksums


def test_non_dry_run_refuses_source_dataset_output(tmp_path: Path) -> None:
    dataset_root, filter_path = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="read-only source dataset"):
        main(
            [
                "--dataset-root",
                str(dataset_root),
                "--filter-path",
                str(filter_path),
                "--output-dir",
                str(dataset_root / "success" / "derived"),
                "--no-check-expected",
            ]
        )


def test_non_dry_run_refuses_to_overwrite_f0_without_resume(tmp_path: Path) -> None:
    dataset_root, filter_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "derived"
    args = [
        "--dataset-root",
        str(dataset_root),
        "--filter-path",
        str(filter_path),
        "--output-dir",
        str(output_dir),
        "--no-check-expected",
    ]
    assert main(args) == 0
    with pytest.raises(FileExistsError, match="--resume"):
        main(args)


def test_resume_reuses_complete_output_with_same_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_root, filter_path = _make_dataset(tmp_path)
    output_dir = tmp_path / "derived"
    args = [
        "--dataset-root",
        str(dataset_root),
        "--filter-path",
        str(filter_path),
        "--output-dir",
        str(output_dir),
        "--no-check-expected",
    ]
    assert main(args) == 0
    capsys.readouterr()
    records_mtime = (output_dir / "f0_episodes.jsonl").stat().st_mtime_ns

    assert main([*args, "--resume"]) == 0

    assert (output_dir / "f0_episodes.jsonl").stat().st_mtime_ns == records_mtime
    assert json.loads(capsys.readouterr().out)["input_fingerprint"]
