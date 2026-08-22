# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

from tools.droid.validate_droid_eef_training import validate_topology

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "examples/toml/sft_config"


@pytest.mark.parametrize(
    ("filename", "world_size", "grad_accum", "global_batch"),
    (
        ("action_policy_droid_eef_edge_4gpu.toml", 4, 8, 32),
        ("action_policy_droid_eef_edge_8gpu.toml", 8, 8, 64),
        ("action_policy_droid_eef_edge_16gpu.toml", 16, 8, 128),
        ("action_policy_droid_eef_edge_32gpu.toml", 32, 8, 256),
        ("action_policy_droid_eef_edge_64gpu.toml", 64, 8, 512),
    ),
)
def test_topology_scales_global_batch_with_world_size(
    filename: str, world_size: int, grad_accum: int, global_batch: int
) -> None:
    result = validate_topology(CONFIG_ROOT / filename, world_size)
    assert result == {
        "world_size": world_size,
        "per_rank_batch": 1,
        "grad_accum": grad_accum,
        "global_batch": global_batch,
    }


def test_topology_rejects_wrong_launch_size() -> None:
    with pytest.raises(ValueError, match="topology has 16 ranks"):
        validate_topology(CONFIG_ROOT / "action_policy_droid_eef_edge_16gpu.toml", 32)
