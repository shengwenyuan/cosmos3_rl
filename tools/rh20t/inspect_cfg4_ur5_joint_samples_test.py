# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import numpy as np
import pytest

from tools.rh20t.inspect_cfg4_ur5_joint_samples import ur5_forward_kinematics

pytestmark = pytest.mark.level(0)


def test_ur5_fk_zero_configuration_is_rigid() -> None:
    transform = ur5_forward_kinematics(np.zeros(6))
    np.testing.assert_allclose(transform[3], [0, 0, 0, 1])
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)
