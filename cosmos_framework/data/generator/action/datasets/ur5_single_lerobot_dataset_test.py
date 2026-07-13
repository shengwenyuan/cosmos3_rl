# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

import cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset as ur5_dataset

pytestmark = pytest.mark.level(0)

_ACTION_LAYOUT = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper_close",
)
_CAMERAS = (
    "observation.images.wrist_cam",
    "observation.images.over_shoulder_left_camera",
    "observation.images.over_shoulder_right_camera",
)


def _source(schema="flat_joint_gripper_v1", **overrides):
    values = dict(
        root="/tmp/source",
        name="test_source",
        condition_source="action_t0",
        joint_action_feature="action",
        gripper_action_feature=None,
        camera_features=dict(zip(("primary", "aux_left", "aux_right"), _CAMERAS, strict=True)),
        action_layout=_ACTION_LAYOUT,
        gripper_semantics="close_fraction",
        view_description="Explicit three-view UR5 source used by the unit test.",
    )
    if schema == "flat_joint_gripper_v1":
        pass
    elif schema == "split_joint_gripper_v1":
        values.update(
            condition_source="observation_state_t0",
            joint_action_feature="action.arm_left_joint",
            gripper_action_feature="action.gripper_left",
            joint_state_feature="observation.state.arm_left_joint",
            gripper_state_feature="observation.state.gripper_left",
        )
    else:
        raise AssertionError(schema)
    values.update(overrides)
    return ur5_dataset.UR5SingleSourceSpec(**values)


def _meta(features):
    for camera in _CAMERAS:
        features[camera] = {"dtype": "video", "shape": [720, 1280, 3], "info": {"video.fps": 15.0}}
    return SimpleNamespace(
        fps=15.0,
        info={"features": features},
        tasks=pd.DataFrame({"task_index": [0], "task": ["move A"]}),
    )


def test_flat_schema_validates_layout_and_uses_action_t0():
    features = {"action": {"dtype": "float32", "shape": [7], "names": {"motors": _ACTION_LAYOUT}}}
    ur5_dataset._validate_source(_meta(features), _source(), fps=15.0)

    wrong_layout = (*reversed(_ACTION_LAYOUT),)
    with pytest.raises(ValueError, match="layout mismatch"):
        ur5_dataset._validate_source(_meta(features), _source(action_layout=wrong_layout), fps=15.0)

    action = torch.arange(33 * 7, dtype=torch.float32).reshape(33, 7)
    source = _source()
    result = ur5_dataset._assemble_action(source, {"action": action}, chunk_length=32)
    torch.testing.assert_close(result, action)
    assert ur5_dataset._delta_timestamps(source, fps=15.0, chunk_length=32) == {
        "action": [index / 15.0 for index in range(33)]
    }


def test_split_schema_builds_state_condition_and_targets():
    features = {
        "action.arm_left_joint": {"dtype": "float32", "shape": [6]},
        "action.gripper_left": {"dtype": "float32", "shape": [1]},
        "observation.state.arm_left_joint": {"dtype": "float32", "shape": [6]},
        "observation.state.gripper_left": {"dtype": "float32", "shape": [1]},
    }
    ur5_dataset._validate_source(_meta(features), _source("split_joint_gripper_v1"), fps=15.0)
    sample = {
        "observation.state.arm_left_joint": torch.arange(6, dtype=torch.float32).reshape(1, 6),
        # LeRobot squeezes metadata shape=[1] scalar sequences to [T].
        "observation.state.gripper_left": torch.tensor([0.25]),
        "action.arm_left_joint": torch.arange(32 * 6, dtype=torch.float32).reshape(32, 6),
        "action.gripper_left": torch.linspace(0.0, 1.0, 32),
    }
    source = _source("split_joint_gripper_v1")
    result = ur5_dataset._assemble_action(source, sample, chunk_length=32)

    assert result.shape == (33, 7)
    torch.testing.assert_close(result[0], torch.tensor([0, 1, 2, 3, 4, 5, 0.25]))
    torch.testing.assert_close(result[1:, :6], sample["action.arm_left_joint"])
    torch.testing.assert_close(result[1:, 6], sample["action.gripper_left"])
    timestamps = ur5_dataset._delta_timestamps(source, fps=15.0, chunk_length=32)
    assert timestamps["observation.state.arm_left_joint"] == [0.0]
    assert timestamps["action.arm_left_joint"] == [index / 15.0 for index in range(1, 33)]


@pytest.mark.parametrize("semantics", ["close_fraction", "open_fraction"])
def test_gripper_conversion_to_canonical_close_fraction_preserves_joints(semantics):
    action = torch.zeros(33, 7)
    action[:, -1] = torch.linspace(0.0, 1.0, 33)
    result = ur5_dataset._assemble_action(_source(gripper_semantics=semantics), {"action": action}, chunk_length=32)
    expected = 1.0 - action[:, -1] if semantics == "open_fraction" else action[:, -1]
    torch.testing.assert_close(result[:, -1], expected)
    torch.testing.assert_close(result[:, :6], action[:, :6])


def test_task_lookup_supports_explicit_and_indexed_metadata():
    explicit = SimpleNamespace(tasks=pd.DataFrame({"task_index": [0, 1], "task": ["move A", "move B"]}))
    indexed = SimpleNamespace(tasks=pd.DataFrame({"task_index": [0, 1]}, index=["move A", "move B"]))
    assert ur5_dataset._task_lookup(explicit) == {0: "move A", 1: "move B"}
    assert ur5_dataset._task_lookup(indexed) == {0: "move A", 1: "move B"}


def test_public_factory_forwards_source_contract(monkeypatch):
    captured = {}

    class FakeDataset(Dataset):
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            raise AssertionError(idx)

        def get_shuffle_blocks(self):
            return [(0, 1)]

    monkeypatch.setattr(ur5_dataset, "UR5SingleLeRobotDataset", FakeDataset)
    source = _source()
    ur5_dataset.get_action_ur5_single_sft_dataset(
        sources=[source], sample_stride=8, tokenizer_config=None, iterable_shuffle=False
    )
    assert captured["sources"] == [source]
    assert captured["sample_stride"] == 8
