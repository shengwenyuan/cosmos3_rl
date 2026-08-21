# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np

from tools.droid.compute_eef_action_stats import action_stats, window_actions
from tools.droid.fx4_f2 import F2Config, build_trajectory


def _trajectory(length: int = 33, close_fraction: float = 0.0):
    state = np.zeros((length, 6))
    return build_trajectory(
        timestamp=np.arange(length) / 15.0,
        frame_index=np.arange(length),
        state=state,
        gripper=np.full(length, close_fraction),
        config=F2Config(),
    )


def test_identity_trajectory_has_identity_rot6d_and_open_gripper() -> None:
    action = window_actions(_trajectory(close_fraction=0.0), 0)
    np.testing.assert_allclose(action[:, :3], 0.0)
    expected_rot6d = np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (32, 1))
    np.testing.assert_allclose(action[:, 3:9], expected_rot6d)
    np.testing.assert_allclose(action[:, 9], 1.0)


def test_gripper_source_close_fraction_is_converted_at_future_timestamp() -> None:
    trajectory = _trajectory()
    trajectory.gripper_close_fraction[1:] = 1.0
    action = window_actions(trajectory, 0)
    np.testing.assert_allclose(action[:, 9], 0.0)


def test_relative_translation_is_expressed_in_current_eef_frame() -> None:
    length = 33
    state = np.zeros((length, 6))
    state[:, 5] = np.pi / 2
    state[1:, 1] = 0.1
    trajectory = build_trajectory(
        timestamp=np.arange(length) / 15.0,
        frame_index=np.arange(length),
        state=state,
        gripper=np.zeros(length),
        config=F2Config(),
    )
    action = window_actions(trajectory, 0)
    np.testing.assert_allclose(action[-1, :3], [0.1, 0.0, 0.0], atol=1e-6)


def test_stats_contains_required_finite_fields() -> None:
    stats = action_stats(window_actions(_trajectory(), 0))
    assert stats["count"] == 32
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        assert len(stats[key]) == 10
        assert np.isfinite(stats[key]).all()
