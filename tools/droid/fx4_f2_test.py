# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.droid.fx4_f2 import (
    F2Config,
    build_trajectory,
    motion_gate_reasons,
    range_frame_bounds,
    rotation_step_degrees,
    sample_freeze_ratio,
    smooth_se3,
    trajectory_reason_codes,
)


def _trajectory(length: int = 40):
    timestamp = np.arange(length) / 15.0
    frame_index = np.arange(length)
    state = np.zeros((length, 6))
    state[:, 0] = np.linspace(0.0, 0.1, length)
    state[:, 5] = np.linspace(3.0, 3.3, length)
    return build_trajectory(
        timestamp=timestamp,
        frame_index=frame_index,
        state=state,
        gripper=np.zeros(length),
        config=F2Config(),
    )


def test_se3_smoothing_preserves_endpoints_and_rotation_manifold() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.02, 0.0], [0.2, 0.0, 0.0]])
    rotation = Rotation.from_euler("z", np.array([179.0, -179.0, -177.0])[:, None], degrees=True)
    smooth_xyz, smooth_rotation = smooth_se3(xyz, rotation, window=3)

    np.testing.assert_allclose(smooth_xyz[[0, -1]], xyz[[0, -1]])
    np.testing.assert_allclose(smooth_rotation.as_matrix()[[0, -1]], rotation.as_matrix()[[0, -1]])
    assert np.isfinite(smooth_rotation.as_quat()).all()


def test_rotation_step_handles_euler_wrap_on_manifold() -> None:
    rotation = Rotation.from_euler("z", np.array([179.0, -179.0])[:, None], degrees=True)
    np.testing.assert_allclose(rotation_step_degrees(rotation), [2.0], atol=1e-8)


def test_trajectory_time_gate_accepts_original_15hz() -> None:
    assert trajectory_reason_codes(_trajectory(), F2Config()) == []


def test_trajectory_time_gate_rejects_non_monotonic_timestamp() -> None:
    trajectory = _trajectory()
    timestamp = trajectory.timestamp.copy()
    timestamp[5] = timestamp[4]
    broken = trajectory.__class__(**{**trajectory.__dict__, "timestamp": timestamp})
    reasons = trajectory_reason_codes(broken, F2Config())
    assert "timestamp_not_monotonic" in reasons
    assert "timestamp_not_15hz" in reasons


def test_range_bounds_preserve_complete_32_step_window() -> None:
    assert range_frame_bounds({"clipped_start": 4, "clipped_end": 10, "window_count": 6}, 42, 32) == (4, 42)
    with pytest.raises(ValueError, match="Invalid trainable range"):
        range_frame_bounds({"clipped_start": 4, "clipped_end": 11, "window_count": 7}, 42, 32)


def test_absolute_motion_gate_rejects_only_explicit_limits() -> None:
    metrics = {
        "max_step_translation_m": 0.101,
        "max_step_rotation_deg": 10.0,
        "max_smoothing_translation_m": 0.001,
        "max_smoothing_rotation_deg": 1.0,
    }
    assert motion_gate_reasons(metrics, F2Config()) == ["step_translation_above_absolute_limit"]


def test_sample_freeze_ratio_uses_adjacent_thumbnail_mad() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8) + 100
    samples = [{"status": "ok", "thumbnail": image.copy()} for _ in range(5)]
    assert sample_freeze_ratio(samples, F2Config()) == 1.0
