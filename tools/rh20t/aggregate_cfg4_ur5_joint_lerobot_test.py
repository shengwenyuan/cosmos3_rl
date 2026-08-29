# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

from tools.rh20t.aggregate_cfg4_ur5_joint_lerobot import discover_shards

pytestmark = pytest.mark.level(0)


def test_discover_shards_requires_complete_ordered_set(tmp_path: Path) -> None:
    (tmp_path / "shard-00.staging").mkdir()
    (tmp_path / "shard-01.staging").mkdir()
    assert [path.name for path in discover_shards(tmp_path, 2)] == ["shard-00.staging", "shard-01.staging"]
    with pytest.raises(ValueError, match="mismatch"):
        discover_shards(tmp_path, 3)
