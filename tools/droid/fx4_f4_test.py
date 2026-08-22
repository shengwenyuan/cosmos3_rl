# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from tools.droid.fx4_f4 import POSITIVE_FAMILIES, REJECTION_BUCKETS, deterministic_qc_selection, evaluate_review_gate


def _accepted_records():
    records = []
    episode_index = 0
    for family in POSITIVE_FAMILIES:
        for _ in range(60):
            records.append({"episode_index": episode_index, "task_family": family, "tasks_raw": [family]})
            episode_index += 1
    return records


def _rejected_records():
    records = []
    episode_index = 1000
    for reason in REJECTION_BUCKETS:
        for _ in range(20):
            records.append(
                {
                    "episode_index": episode_index,
                    "task_family": None,
                    "tasks_raw": [reason],
                    "f1_status": "rejected",
                    "f1_reason_codes": [reason],
                }
            )
            episode_index += 1
    return records


def test_qc_selection_has_exact_320_unique_coverage() -> None:
    selected = deterministic_qc_selection(accepted=_accepted_records(), rejected=_rejected_records())
    assert len(selected) == 320
    assert len({item["episode_index"] for item in selected}) == 320
    for family in POSITIVE_FAMILIES:
        assert sum(item["bucket"] == f"accepted:{family}" for item in selected) == 50
    for reason in REJECTION_BUCKETS:
        assert sum(item["bucket"] == reason for item in selected) == 15
    assert sum(item["bucket"] == "random" for item in selected) == 30


def test_review_gate_is_pending_below_minimum_sample() -> None:
    rows = [{"bucket": "accepted:pick_place_relocate", "review_label": "correct"}] * 31
    assert evaluate_review_gate(rows)["status"] == "pending"


def test_review_gate_accepts_partial_sample_and_enforces_five_percent() -> None:
    rows = [{"bucket": "accepted:pick_place_relocate", "review_label": "correct"} for _ in range(32)]
    rows.extend({"bucket": "random", "review_label": ""} for _ in range(288))
    assert evaluate_review_gate(rows)["status"] == "pass"
    assert evaluate_review_gate(rows)["pending_rows"] == 288
    rows[0]["review_label"] = "incorrect"
    assert evaluate_review_gate(rows)["status"] == "pass"
    rows[1]["review_label"] = "incorrect"
    assert evaluate_review_gate(rows)["status"] == "fail"
