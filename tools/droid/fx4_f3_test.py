# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np

from tools.droid.fx4_f2 import F2Config, build_trajectory
from tools.droid.fx4_f3 import (
    F3Config,
    assign_group_splits,
    candidate_window_starts,
    cap_episode_candidates,
    scene_group_id,
)


def _trajectory(length: int = 80):
    state = np.zeros((length, 6))
    state[:, 0] = np.arange(length) * 0.001
    gripper = np.zeros(length)
    gripper[20:] = 0.6
    return build_trajectory(
        timestamp=np.arange(length) / 15.0,
        frame_index=np.arange(length),
        state=state,
        gripper=gripper,
        config=F2Config(),
    )


def test_candidate_starts_are_deterministic_event_union() -> None:
    config = F3Config(translation_m=0.005, rotation_deg=3.0, max_gap_frames=12)
    left = candidate_window_starts(_trajectory(), start=0, stop=48, config=config, max_gap_frames=12)
    right = candidate_window_starts(_trajectory(), start=0, stop=48, config=config, max_gap_frames=12)
    assert left == right
    assert left[0] == {"start": 0, "events": ["range_start"]}
    assert left[-1]["start"] == 47
    assert any("translation" in item["events"] for item in left)
    assert any("gripper_crossing" in item["events"] for item in left)


def test_force_unavailable_path_still_obeys_gap() -> None:
    trajectory = _trajectory()
    trajectory.raw_xyz[:] = 0.0
    trajectory.smooth_xyz[:] = 0.0
    trajectory.gripper_close_fraction[:] = 0.0
    starts = candidate_window_starts(trajectory, start=0, stop=34, config=F3Config(), max_gap_frames=12)
    assert [item["start"] for item in starts] == [0, 12, 24, 33]


def test_gap_probe_changes_candidate_density() -> None:
    trajectory = _trajectory()
    trajectory.raw_xyz[:] = 0.0
    trajectory.smooth_xyz[:] = 0.0
    trajectory.gripper_close_fraction[:] = 0.0
    config = F3Config()
    counts = {
        gap: len(candidate_window_starts(trajectory, start=0, stop=48, config=config, max_gap_frames=gap))
        for gap in (8, 12, 16)
    }
    assert counts[8] > counts[12] > counts[16]


def test_episode_cap_keeps_each_range_endpoints() -> None:
    candidates = [
        [{"start": index, "events": []} for index in range(40)],
        [{"start": 100 + index, "events": []} for index in range(40)],
    ]
    selected = cap_episode_candidates(candidates, 64)
    assert len(selected) == 64
    assert {(0, 0), (0, 39), (1, 100), (1, 139)} <= selected


def test_scene_group_uses_institution_and_collection_date() -> None:
    assert scene_group_id("AUTOLab/success/2023-07-07/Fri_Jul") == "AUTOLab/2023-07-07"


def test_group_split_is_deterministic_and_group_atomic() -> None:
    records = [
        {"group_id": "lab/day1", "task_family": "pick_place_relocate"},
        {"group_id": "lab/day1", "task_family": "pick_place_relocate"},
        {"group_id": "lab/day2", "task_family": "pick_place_relocate"},
    ]
    assert assign_group_splits(records, F3Config()) == assign_group_splits(records, F3Config())
