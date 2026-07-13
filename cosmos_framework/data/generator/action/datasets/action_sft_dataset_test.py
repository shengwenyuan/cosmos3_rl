# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionSFTDataset,
    _validate_droid_action_policy_manifest,
)
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset_config import (
    COSMOS3_DROID_SUCCESS_PROFILE,
    LEROBOT_ROOTS,
    SOURCE_GRIPPER_SEMANTICS,
)
from cosmos_framework.data.generator.action.policy_schema import ActionPolicyManifest, load_action_policy_manifest

pytestmark = pytest.mark.level(0)


def test_map_wrapper_forwards_source_action_normalizer() -> None:
    normalizer = object()
    sample = {"mode": "policy"}
    captured = {}

    class Source(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return sample

        def get_action_normalizer(self, value):
            assert value is sample
            return normalizer

    def transform(value, resolution, action_normalizer=None):
        captured.update(value=value, resolution=resolution, action_normalizer=action_normalizer)
        return value

    assert ActionSFTDataset(Source(), transform, "480")[0] is sample
    assert captured == {"value": sample, "resolution": "480", "action_normalizer": normalizer}


def test_public_droid_profile_has_explicit_source_and_model_gripper_semantics() -> None:
    dataset = object.__new__(DROIDLeRobotDataset)
    dataset._source_gripper_semantics = SOURCE_GRIPPER_SEMANTICS[COSMOS3_DROID_SUCCESS_PROFILE]
    source_close = torch.tensor([0.0, 0.25, 1.0])

    assert LEROBOT_ROOTS[COSMOS3_DROID_SUCCESS_PROFILE] is None
    torch.testing.assert_close(dataset._gripper_to_model_semantics(source_close), 1.0 - source_close)


def test_public_droid_profile_rejects_manifest_semantic_drift() -> None:
    manifest = load_action_policy_manifest(
        Path(__file__).parents[5] / "examples/toml/sft_config/action_policy_droid_repro.toml"
    )
    config = SimpleNamespace(
        dataset_profile=COSMOS3_DROID_SUCCESS_PROFILE,
        view_description=manifest.datasets[0].view_description,
    )
    _validate_droid_action_policy_manifest(config, manifest)

    raw = manifest.model_dump(mode="json")
    raw["domain_name"] = "ur5-single-joint"
    with pytest.raises(ValueError, match="domain_name"):
        _validate_droid_action_policy_manifest(config, ActionPolicyManifest.model_validate(raw))

    raw = manifest.model_dump(mode="json")
    swapped = ["joint_1", "joint_0", *(f"joint_{index}" for index in range(2, 7)), "gripper"]
    raw["model_action"]["layout"] = swapped
    raw["wire_action"]["layout"] = swapped
    with pytest.raises(ValueError, match="model_layout"):
        _validate_droid_action_policy_manifest(config, ActionPolicyManifest.model_validate(raw))
