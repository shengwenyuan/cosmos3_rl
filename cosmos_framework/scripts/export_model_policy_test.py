# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

from cosmos_framework.data.generator.action.policy_schema import (
    load_action_policy_manifest,
    save_action_policy_manifest,
)
from cosmos_framework.scripts.export_model import (
    _resolve_action_policy_manifest,
    _validate_action_policy_destination,
)

pytestmark = pytest.mark.level(0)


def test_export_rejects_explicit_manifest_that_conflicts_with_checkpoint_owner(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    owned = load_action_policy_manifest(repo / "examples/toml/sft_config/action_policy_droid_repro.toml")
    conflicting = repo / "examples/toml/sft_config/action_policy_ur5_single_joint_overfit.toml"
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_1" / "model"
    checkpoint.mkdir(parents=True)
    save_action_policy_manifest(owned, tmp_path / "run" / "action_policy.yaml")

    with pytest.raises(ValueError, match="conflicts with"):
        _resolve_action_policy_manifest(str(checkpoint), conflicting)


def test_export_rejects_stale_destination_before_writing(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    source = load_action_policy_manifest(repo / "examples/toml/sft_config/action_policy_droid_repro.toml")
    stale = load_action_policy_manifest(repo / "examples/toml/sft_config/action_policy_ur5_single_joint_overfit.toml")
    output = tmp_path / "export"
    save_action_policy_manifest(stale, output / "action_policy.yaml")

    with pytest.raises(ValueError, match="different exported"):
        _validate_action_policy_destination(source, output)
