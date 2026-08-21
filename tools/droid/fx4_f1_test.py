# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

from tools.droid.fx4_f1 import classify_episode, load_taxonomy, parse_task_annotations


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(Path(__file__).with_name("fx4_f1_taxonomy.yaml"))


def test_parse_task_annotations_flattens_lerobot_list() -> None:
    assert parse_task_annotations(["Pick up the cup | Put the cup down | Move the cup"]) == [
        "pick up the cup",
        "put the cup down",
        "move the cup",
    ]


def test_two_vote_pick_place_is_accepted(taxonomy) -> None:
    result = classify_episode(
        length=120,
        f0_window_count=88,
        tasks_value=["Pick up the cup | Put the cup on the table | Relocate the cup"],
        taxonomy=taxonomy,
    )
    assert result["f1_status"] == "accepted"
    assert result["task_family"] == "pick_place_relocate"


def test_pouring_is_hard_excluded_even_with_place_vote(taxonomy) -> None:
    result = classify_episode(
        length=120,
        f0_window_count=88,
        tasks_value=["Pour the contents into the bowl | Put the cup in the bowl | Pour water into the bowl"],
        taxonomy=taxonomy,
    )
    assert result["f1_status"] == "rejected"
    assert "hard_exclude:liquid_pouring" in result["f1_reason_codes"]


def test_stack_arrange_wins_family_precedence(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Stack the blocks | Pick and stack the cubes | Arrange the blocks in a row"],
        taxonomy=taxonomy,
    )
    assert result["f1_status"] == "accepted"
    assert result["task_family"] == "stack_arrange"


def test_cloth_task_is_hard_excluded(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Move the towel | Put the cloth in the bowl | Pick up the towel"],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:cloth_bag_fold" in result["f1_reason_codes"]


def test_non_stack_multistage_is_rejected(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Open the drawer and put the cup in it | Open the drawer then place the cup | Move the cup"],
        taxonomy=taxonomy,
    )
    assert "complex_multistage" in result["f1_reason_codes"]


def test_relocating_tool_object_is_not_tool_operation(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=[
            "Pick up the screwdriver and place it in the drawer",
            "Put the screw driver in the drawer",
            "Move the screwdriver into the drawer",
        ],
        taxonomy=taxonomy,
    )
    assert result["f1_status"] == "accepted"
    assert result["task_family"] == "pick_place_relocate"


def test_using_cutting_tool_is_rejected(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Use the knife on the fruit", "Cut the fruit", "Slice the fruit"],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:cutting_tool" in result["f1_reason_codes"]


@pytest.mark.parametrize(
    "annotation",
    ["Put seasoning on the slice of bread", "Move the box with shape cut outs"],
)
def test_cutting_noun_is_not_tool_operation(taxonomy, annotation: str) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=[annotation, "Move the object", "Put the object on the table"],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:cutting_tool" not in result["f1_reason_codes"]


def test_screw_fastener_operation_is_rejected(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Screw the cap on", "Tighten the cap", "Screw it onto the bottle"],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:insertion_fastener" in result["f1_reason_codes"]


def test_zipper_on_clothing_is_rejected(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Pull the zipper up", "Close the zip on the jersey", "Pull the vest zipper"],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:cloth_bag_fold" in result["f1_reason_codes"]


def test_connect_four_is_rejected_as_insertion(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=[
            "Put the orange disc in the blue board",
            "Put the orange ring in the rack",
            "Put the orange chip into the connect four game",
        ],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:insertion_fastener" in result["f1_reason_codes"]


def test_loosen_rope_is_not_fastener_reason(taxonomy) -> None:
    result = classify_episode(
        length=180,
        f0_window_count=148,
        tasks_value=["Loosen the rope knot", "Move the rope", "Untangle the string"],
        taxonomy=taxonomy,
    )
    assert "hard_exclude:insertion_fastener" not in result["f1_reason_codes"]
    assert "hard_exclude:cloth_bag_fold" in result["f1_reason_codes"]


@pytest.mark.parametrize(
    ("length", "reason"),
    [(47, "length_below_min"), (451, "length_above_max")],
)
def test_length_gate(taxonomy, length: int, reason: str) -> None:
    result = classify_episode(
        length=length,
        f0_window_count=10,
        tasks_value=["Move the cup | Move the mug | Shift the cup"],
        taxonomy=taxonomy,
    )
    assert reason in result["f1_reason_codes"]


def test_f0_invalid_episode_cannot_enter_f1(taxonomy) -> None:
    result = classify_episode(
        length=120,
        f0_window_count=0,
        tasks_value=["Move the cup | Move the mug | Shift the cup"],
        taxonomy=taxonomy,
    )
    assert "f0_not_trainable" in result["f1_reason_codes"]
