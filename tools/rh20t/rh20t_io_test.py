# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest

from tools.rh20t.rh20t_io import (
    close_fraction,
    contiguous_true_runs,
    deterministic_split,
    interpolation_valid,
    nearest_indices,
    parse_episode_name,
)

pytestmark = pytest.mark.level(0)


def test_parse_episode_name_excludes_all_human_suffixes() -> None:
    robot = parse_episode_name("task_0034_user_0017_scene_0004_cfg_0004")
    human = parse_episode_name("task_0085_user_0007_scene_0010_cfg_0004_human_2")
    assert (robot.task_id, robot.scene_number, robot.cfg, robot.is_human) == ("task_0034", 4, 4, False)
    assert human.is_human
    with pytest.raises(ValueError, match="Unsupported"):
        parse_episode_name("task_0034_bad")


def test_alignment_helpers_do_not_bridge_large_gaps() -> None:
    source = np.asarray([0.0, 100.0, 400.0, 500.0])
    target = np.asarray([50.0, 150.0, 350.0, 450.0])
    indices, errors = nearest_indices(source, target)
    np.testing.assert_array_equal(indices, [0, 1, 2, 2])
    np.testing.assert_array_equal(errors, [50.0, 50.0, 50.0, 50.0])
    np.testing.assert_array_equal(interpolation_valid(source, target, 150.0), [True, False, False, True])
    assert contiguous_true_runs(np.asarray([1, 1, 0, 1, 1, 1], dtype=bool), min_length=2) == [(0, 2), (3, 6)]


def test_gripper_width_converts_to_close_fraction() -> None:
    np.testing.assert_allclose(close_fraction(np.asarray([0.0, 42.5, 85.0, 100.0])), [1.0, 0.5, 0.0, 0.0])


def test_split_is_reproducible() -> None:
    first = deterministic_split("task|scene|user", 42, 0.9, 0.05)
    assert deterministic_split("task|scene|user", 42, 0.9, 0.05) == first
