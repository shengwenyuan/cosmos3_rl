# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.rh20t.convert_cfg4_ur5_joint_to_lerobot import _resample_segment
from tools.rh20t.rh20t_io import EXTERIOR_CAMERAS, PRIMARY_SERIAL

pytestmark = pytest.mark.level(0)


def test_resample_segment_builds_absolute_joint_and_close_fraction(tmp_path: Path) -> None:
    episode = tmp_path / "task_0001_user_0010_scene_0001_cfg_0004"
    transformed = episode / "transformed"
    transformed.mkdir(parents=True)
    timestamps = np.arange(0, 3100, 100, dtype=np.int64)
    joint_values = np.stack([np.linspace(0.0, 0.5, 6) + timestamp / 10000 for timestamp in timestamps])
    np.save(
        transformed / "joint.npy",
        {PRIMARY_SERIAL: {int(timestamp): value for timestamp, value in zip(timestamps, joint_values, strict=True)}},
    )
    np.save(
        transformed / "gripper.npy",
        {
            PRIMARY_SERIAL: {
                int(timestamp): {"gripper_command": [85.0 - timestamp / 100, 77, 200], "gripper_info": [84, 0, 0]}
                for timestamp in timestamps
            }
        },
    )
    for serial in EXTERIOR_CAMERAS:
        camera = episode / f"cam_{serial}"
        camera.mkdir()
        np.save(camera / "timestamps.npy", {"color": timestamps})

    record = {
        "source_path": str(episode),
        "segment_id": "synthetic__segment_00",
        "target_start_ms": 0.0,
        "target_step_ms": 1000 / 15,
        "target_fps": 15,
        "frame_count": 33,
    }
    result = _resample_segment(record)
    assert result["joint"].shape == (33, 6)
    assert result["gripper"].shape == (33,)
    assert result["gripper"][0] == pytest.approx(0.0)
    assert result["gripper"][-1] > result["gripper"][0]
    assert all(error.max() <= 50.0 for error in result["camera_errors"].values())
