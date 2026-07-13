# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import cosmos_framework.data.generator.action.datasets.ur5_single_eef_lerobot_dataset as eef_dataset

pytestmark = pytest.mark.level(0)

_ACTION_LAYOUT = (
    "delta_x",
    "delta_y",
    "delta_z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper",
)
_CAMERAS = {
    "primary": "observation.images.wrist_cam",
    "aux_left": "observation.images.over_shoulder_left_camera",
    "aux_right": "observation.images.over_shoulder_right_camera",
}


def _source(**overrides):
    values = dict(
        root="/tmp/eef-source",
        name="eef_source",
        pose_feature="action.tool0_pose",
        quaternion_order="xyzw",
        gripper_action_feature="action",
        gripper_index=6,
        gripper_target_offset=1,
        eef_frame="tool0",
        camera_features=_CAMERAS,
        action_layout=_ACTION_LAYOUT,
        gripper_semantics="close_fraction",
        view_description="Wrist above, left and right over-shoulder cameras below.",
    )
    values.update(overrides)
    return eef_dataset.UR5SingleEEFSourceSpec(**values)


def _meta():
    features = {
        "action.tool0_pose": {"dtype": "float32", "shape": [7]},
        "action": {"dtype": "float32", "shape": [7]},
    }
    for camera in _CAMERAS.values():
        features[camera] = {"dtype": "video", "shape": [720, 1280, 3], "info": {"video.fps": 15.0}}
    return SimpleNamespace(
        fps=15.0,
        info={"features": features},
        tasks=pd.DataFrame({"task_index": [0], "task": ["pick up A and place it near B"]}),
    )


def _identity_pose(rows: int, *, quaternion_order: str = "xyzw") -> torch.Tensor:
    pose = torch.zeros(rows, 7)
    pose[:, 0] = torch.arange(rows, dtype=torch.float32)
    pose[:, 3 if quaternion_order == "wxyz" else 6] = 1.0
    return pose


def test_source_validation_and_timestamps_align_gripper_with_t_plus_one() -> None:
    source = _source()
    eef_dataset._validate_source(_meta(), source, fps=15.0)
    timestamps = eef_dataset._delta_timestamps(source, fps=15.0, chunk_length=32)
    assert timestamps[source.pose_feature] == [index / 15.0 for index in range(33)]
    assert timestamps[source.gripper_action_feature] == [index / 15.0 for index in range(1, 33)]
    for camera in _CAMERAS.values():
        assert timestamps[camera] == timestamps[source.pose_feature]


def test_assemble_action_has_32_by_10_shape_and_t_plus_one_gripper_targets() -> None:
    source = _source()
    native_action = torch.zeros(32, 7)
    native_action[:, 6] = torch.arange(1, 33, dtype=torch.float32) / 32.0
    result = eef_dataset._assemble_action(
        source,
        {source.pose_feature: _identity_pose(33), source.gripper_action_feature: native_action},
        chunk_length=32,
    )
    assert result.shape == (32, 10)
    torch.testing.assert_close(result[:, 0], torch.ones(32))
    torch.testing.assert_close(result[:, 1:3], torch.zeros(32, 2))
    torch.testing.assert_close(result[:, 3:9], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(32, 1))
    torch.testing.assert_close(result[:, 9], native_action[:, 6])


def test_wxyz_source_decodes_a_known_rotation() -> None:
    source = _source(quaternion_order="wxyz")
    pose = _identity_pose(2, quaternion_order="wxyz")
    half_angle = torch.tensor(torch.pi / 4)
    pose[1, 3:] = torch.tensor([torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)])
    result = eef_dataset._assemble_action(
        source,
        {source.pose_feature: pose, source.gripper_action_feature: torch.zeros(1, 7)},
        chunk_length=1,
    )
    torch.testing.assert_close(result[0, 3:9], torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("semantics", "expected"),
    [
        ("close_fraction", torch.tensor([0.0, 1.0])),
        ("open_fraction", torch.tensor([1.0, 0.0])),
    ],
)
def test_gripper_is_converted_to_close_fraction_exactly_once(semantics, expected) -> None:
    source = _source(gripper_semantics=semantics)
    native_action = torch.zeros(2, 7)
    native_action[:, 6] = torch.tensor([0.0, 1.0])
    result = eef_dataset._assemble_action(
        source,
        {source.pose_feature: _identity_pose(3), source.gripper_action_feature: native_action},
        chunk_length=2,
    )
    torch.testing.assert_close(result[:, 9], expected)


def _manifest():
    source = SimpleNamespace(
        root="/tmp/eef-source",
        name="eef_source",
        condition_source="none",
        action_features=("action.tool0_pose", "action"),
        state_features=(),
        camera_features=_CAMERAS,
        action_layout=_ACTION_LAYOUT,
        gripper_semantics="close_fraction",
        view_description="Wrist above, left and right over-shoulder cameras below.",
        source_quaternion_order="xyzw",
        source_gripper_index=6,
        source_target_offset=1,
        source_frame="tool0",
    )
    return SimpleNamespace(
        robot="ur5",
        domain_name="ur5-single-eef",
        policy_fps=15,
        chunk_size=32,
        model_action=SimpleNamespace(
            codec="eef_delta",
            representation="delta",
            layout=_ACTION_LAYOUT,
            frame="tool0",
            gripper=SimpleNamespace(index=9, semantics="close_fraction"),
        ),
        wire_action=SimpleNamespace(codec="eef_absolute", frame="tool0"),
        conditioning=SimpleNamespace(state_rows=0, history_rows=0, source="none"),
        observation=SimpleNamespace(view_shape_hw=(360, 640)),
        datasets=(source,),
    )


def test_manifest_binder_and_validator_keep_all_source_conventions_explicit() -> None:
    manifest = _manifest()
    sources = eef_dataset._bind_ur5_single_eef_action_policy_manifest(manifest)
    assert sources == [
        {
            "root": "/tmp/eef-source",
            "name": "eef_source",
            "pose_feature": "action.tool0_pose",
            "quaternion_order": "xyzw",
            "gripper_action_feature": "action",
            "gripper_index": 6,
            "gripper_target_offset": 1,
            "eef_frame": "tool0",
            "camera_features": _CAMERAS,
            "action_layout": _ACTION_LAYOUT,
            "gripper_semantics": "close_fraction",
            "view_description": "Wrist above, left and right over-shoulder cameras below.",
            "tolerance_s": 2e-4,
            "decode_size_hw": (360, 640),
        }
    ]
    eef_dataset._validate_ur5_single_eef_action_policy_manifest(SimpleNamespace(sources=sources), manifest)
    assert (
        getattr(eef_dataset.get_action_ur5_single_eef_sft_dataset, "action_policy_manifest_binder")
        is eef_dataset._bind_ur5_single_eef_action_policy_manifest
    )

    drifted = [{**sources[0], "quaternion_order": "wxyz"}]
    with pytest.raises(ValueError, match="quaternion_order"):
        eef_dataset._validate_ur5_single_eef_action_policy_manifest(SimpleNamespace(sources=drifted), manifest)


def test_manifest_rejects_frame_or_target_offset_drift() -> None:
    manifest = _manifest()
    manifest.datasets[0].source_frame = "base"
    with pytest.raises(ValueError, match="frame"):
        eef_dataset._bind_ur5_single_eef_action_policy_manifest(manifest)

    manifest = _manifest()
    manifest.datasets[0].source_target_offset = 0
    with pytest.raises(ValueError, match="source_target_offset=1"):
        eef_dataset._bind_ur5_single_eef_action_policy_manifest(manifest)
