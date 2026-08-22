# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

from tools.droid.finalize_fx4 import build_final_summary


def test_final_summary_remains_pending_until_human_review(tmp_path) -> None:
    summaries = {
        "f0_summary.json": {"expected_counts_match": True},
        "f1_summary.json": {"counts": {}},
        "f2_motion_summary.json": {"gates": {"finite": True}},
        "f2_summary.json": {"gates": {"video_decode_failure_lt_0_001": True}},
        "f3_summary.json": {"gates": {"episode_overlap": 0, "all_families_in_val": True}},
        "f4_summary.json": {"gates": {"coverage_pass": True, "semantic_review": {"status": "pending"}}},
    }
    for filename, value in summaries.items():
        (tmp_path / filename).write_text(json.dumps(value))
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    summary = build_final_summary(tmp_path, stats)
    assert summary["status"] == "pending_human_review"


def test_final_summary_fails_on_false_boolean_gate(tmp_path) -> None:
    summaries = {
        "f0_summary.json": {"expected_counts_match": True},
        "f1_summary.json": {},
        "f2_motion_summary.json": {"gates": {"finite": True}},
        "f2_summary.json": {"gates": {"video_decode_failure_lt_0_001": True}},
        "f3_summary.json": {"gates": {"episode_overlap": 0, "all_families_in_val": False}},
        "f4_summary.json": {"gates": {"coverage_pass": True, "semantic_review": {"status": "pass"}}},
    }
    for filename, value in summaries.items():
        (tmp_path / filename).write_text(json.dumps(value))
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    assert build_final_summary(tmp_path, stats)["status"] == "fail"


def test_final_summary_uses_live_sampled_review(tmp_path) -> None:
    summaries = {
        "f0_summary.json": {"expected_counts_match": True},
        "f1_summary.json": {"counts": {}},
        "f2_motion_summary.json": {"gates": {"finite": True}},
        "f2_summary.json": {"gates": {"video_decode_failure_lt_0_001": True}},
        "f3_summary.json": {"gates": {"episode_overlap": 0, "all_families_in_val": True}},
        "f4_summary.json": {"gates": {"coverage_pass": True, "semantic_review": {"status": "pending"}}},
    }
    for filename, value in summaries.items():
        (tmp_path / filename).write_text(json.dumps(value))
    review_rows = "\n".join(f"{index},accepted:open_close,correct" for index in range(32))
    (tmp_path / "qc_review.csv").write_text(f"episode_index,bucket,review_label\n{review_rows}\n")
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    summary = build_final_summary(tmp_path, stats)
    assert summary["status"] == "pass"
    assert summary["f4_semantic_review"]["decision_mode"] == "sampled_review"
