# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Source-neutral LeRobot adapter for UR5 EEF-delta policies.

Each source declares its stored absolute-pose and gripper conventions.  The
adapter samples 33 absolute poses at ``t0..t32`` and 32 gripper targets at the
corresponding next-pose timestamps, then emits the canonical model action::

    [delta_translation(3), delta_rotation_rot6d(6), gripper_close_fraction(1)]

No storage field name, quaternion order, gripper column, EEF frame, or camera
mapping is inferred from a dataset name.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.datasets.canvas_utils import concat_three_view_canvas, resize_view
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    BaseActionLeRobotDataset,
)
from cosmos_framework.data.generator.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

QuaternionOrder = Literal["xyzw", "wxyz"]
GripperSemantics = Literal["close_fraction", "open_fraction"]

_ACTION_DIM = 10
_POSE_DIM = 7
_CAMERA_ROLES = ("primary", "aux_left", "aux_right")
_EEF_DELTA_LAYOUT = (
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
_DEFAULT_TOLERANCE_S = 2e-4


@dataclass(frozen=True, slots=True)
class UR5SingleEEFSourceSpec:
    """Explicitly map one LeRobot source into the canonical UR5 EEF policy."""

    root: str
    name: str
    pose_feature: str
    quaternion_order: QuaternionOrder
    gripper_action_feature: str
    gripper_index: int
    gripper_target_offset: int
    eef_frame: str
    camera_features: Mapping[str, str]
    action_layout: tuple[str, ...]
    gripper_semantics: GripperSemantics
    view_description: str
    tolerance_s: float = _DEFAULT_TOLERANCE_S
    decode_size_hw: tuple[int, int] | None = (360, 640)


def _normalize_source(source: UR5SingleEEFSourceSpec | Mapping[str, Any]) -> UR5SingleEEFSourceSpec:
    if isinstance(source, UR5SingleEEFSourceSpec):
        return source
    values = dict(source)
    for key in ("action_layout", "decode_size_hw"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    if values.get("camera_features") is not None:
        values["camera_features"] = dict(values["camera_features"])
    return UR5SingleEEFSourceSpec(**values)


def _require_float_feature(features: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], int]:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise ValueError(f"Missing required UR5-single EEF feature {key!r}.")
    shape = tuple(feature.get("shape", ()))
    if feature.get("dtype") != "float32" or len(shape) != 1 or int(shape[0]) <= 0:
        raise ValueError(f"Feature {key!r} must be float32 with one positive-width dimension, got {dict(feature)!r}.")
    return feature, int(shape[0])


def _task_lookup(meta: LeRobotDatasetMetadata) -> dict[int, str]:
    tasks = meta.tasks
    columns = set(getattr(tasks, "columns", ()))
    if "task" in columns:
        result = {
            int(row["task_index"]) if "task_index" in columns else int(row_index): str(row["task"])
            for row_index, row in tasks.iterrows()
        }
    else:
        result = {task_index: str(tasks.iloc[task_index].name) for task_index in range(len(tasks))}
    if not result or any(not task.strip() for task in result.values()):
        raise ValueError("UR5-single EEF source metadata contains no usable task text.")
    return result


def _validate_source(meta: LeRobotDatasetMetadata, source: UR5SingleEEFSourceSpec, *, fps: float) -> None:
    if not source.root.strip() or not source.name.strip() or not source.view_description.strip():
        raise ValueError("UR5-single EEF source root, name, and view_description must not be empty.")
    if not source.pose_feature.strip() or not source.gripper_action_feature.strip() or not source.eef_frame.strip():
        raise ValueError("UR5-single EEF pose/gripper feature names and eef_frame must not be empty.")
    if source.pose_feature == source.gripper_action_feature:
        raise ValueError("UR5-single EEF pose_feature and gripper_action_feature must be distinct.")
    if source.quaternion_order not in ("xyzw", "wxyz"):
        raise ValueError(f"Unsupported source quaternion_order={source.quaternion_order!r}.")
    if source.gripper_target_offset != 1:
        raise ValueError("UR5-single EEF deltas require gripper_target_offset=1 to align each target with t+1.")
    if source.gripper_semantics not in ("close_fraction", "open_fraction"):
        raise ValueError(f"Unsupported source gripper_semantics={source.gripper_semantics!r}.")
    if tuple(source.action_layout) != _EEF_DELTA_LAYOUT:
        raise ValueError(f"UR5-single EEF action_layout must be {_EEF_DELTA_LAYOUT!r}.")
    if set(source.camera_features) != set(_CAMERA_ROLES):
        raise ValueError(f"camera_features must explicitly map all three roles {_CAMERA_ROLES!r}.")
    if len(set(source.camera_features.values())) != len(_CAMERA_ROLES):
        raise ValueError("Each UR5-single EEF camera role must map to a distinct feature.")
    if source.tolerance_s <= 0:
        raise ValueError(f"tolerance_s must be positive, got {source.tolerance_s!r}.")
    if source.decode_size_hw is not None and (
        len(source.decode_size_hw) != 2 or any(int(value) <= 0 for value in source.decode_size_hw)
    ):
        raise ValueError(f"decode_size_hw must contain two positive values, got {source.decode_size_hw!r}.")
    if float(meta.fps) != float(fps):
        raise ValueError(f"Source fps={meta.fps} does not match policy fps={fps}; implicit resampling is forbidden.")

    features = meta.info.get("features", {})
    _, pose_width = _require_float_feature(features, source.pose_feature)
    if pose_width != _POSE_DIM:
        raise ValueError(
            f"Pose feature {source.pose_feature!r} must be [xyz(3), quaternion(4)] with width {_POSE_DIM}, "
            f"got {pose_width}."
        )
    _, gripper_width = _require_float_feature(features, source.gripper_action_feature)
    if source.gripper_index < 0 or source.gripper_index >= gripper_width:
        raise ValueError(
            f"gripper_index={source.gripper_index} is outside feature {source.gripper_action_feature!r} "
            f"width {gripper_width}."
        )

    for camera in source.camera_features.values():
        feature = features.get(camera)
        if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
            raise ValueError(f"Camera feature {camera!r} is missing or is not video.")
        camera_fps = feature.get("info", {}).get("video.fps")
        if camera_fps is not None and float(camera_fps) != float(fps):
            raise ValueError(f"Camera {camera!r} fps={camera_fps} does not match policy fps={fps}.")


def _delta_timestamps(source: UR5SingleEEFSourceSpec, *, fps: float, chunk_length: int) -> dict[str, list[float]]:
    pose_times = [index / fps for index in range(chunk_length + 1)]
    gripper_times = [(index + source.gripper_target_offset) / fps for index in range(chunk_length)]
    timestamps = {
        source.pose_feature: pose_times,
        source.gripper_action_feature: gripper_times,
    }
    timestamps.update({camera: pose_times for camera in source.camera_features.values()})
    return timestamps


def _pose_rows(sample: Mapping[str, Any], source: UR5SingleEEFSourceSpec, rows: int) -> torch.Tensor:
    value = torch.as_tensor(sample[source.pose_feature]).float()
    expected = (rows, _POSE_DIM)
    if tuple(value.shape) != expected:
        raise ValueError(
            f"EEF pose feature {source.pose_feature!r} has shape {tuple(value.shape)}, expected {expected}."
        )
    return value


def _gripper_rows(sample: Mapping[str, Any], source: UR5SingleEEFSourceSpec, rows: int) -> torch.Tensor:
    value = torch.as_tensor(sample[source.gripper_action_feature]).float()
    if value.ndim == 1:
        if tuple(value.shape) != (rows,) or source.gripper_index != 0:
            raise ValueError(
                f"EEF gripper feature {source.gripper_action_feature!r} has shape {tuple(value.shape)} and "
                f"cannot supply index {source.gripper_index} for {rows} rows."
            )
        gripper = value.unsqueeze(-1)
    elif value.ndim == 2 and value.shape[0] == rows and source.gripper_index < value.shape[1]:
        gripper = value[:, source.gripper_index : source.gripper_index + 1]
    else:
        raise ValueError(
            f"EEF gripper feature {source.gripper_action_feature!r} has shape {tuple(value.shape)} and "
            f"cannot supply index {source.gripper_index} for {rows} rows."
        )
    if source.gripper_semantics == "open_fraction":
        gripper = 1.0 - gripper
    return gripper


def _assemble_action(source: UR5SingleEEFSourceSpec, sample: Mapping[str, Any], *, chunk_length: int) -> torch.Tensor:
    pose = _pose_rows(sample, source, chunk_length + 1)
    absolute = build_abs_pose_from_components(
        pose[:, :3],
        pose[:, 3:],
        f"quat_{source.quaternion_order}",
    )
    deltas = pose_abs_to_rel(absolute, rotation_format="rot6d", pose_convention="backward_framewise")
    gripper = _gripper_rows(sample, source, chunk_length)
    action = torch.cat((torch.from_numpy(np.asarray(deltas)).float(), gripper), dim=-1)
    if tuple(action.shape) != (chunk_length, _ACTION_DIM):
        raise ValueError(
            f"UR5-single EEF action has shape {tuple(action.shape)}, expected {(chunk_length, _ACTION_DIM)}."
        )
    return action


class UR5SingleEEFLeRobotDataset(BaseActionLeRobotDataset):
    """Canonical 10-D UR5 EEF policy over one or more explicit sources."""

    def __init__(
        self,
        *,
        sources: Sequence[UR5SingleEEFSourceSpec | Mapping[str, Any]],
        fps: float = 15.0,
        chunk_length: int = 32,
        sample_stride: int = 1,
        split: str = "full",
        split_seed: int = 42,
        split_val_ratio: float = 0.0,
        mode: str = "policy",
        viewpoint: str = "concat_view",
        action_normalization: ActionNormalization | None = None,
        video_backend: str | None = "torchcodec",
        skip_video_loading: bool = False,
    ) -> None:
        if not sources:
            raise ValueError("UR5SingleEEFLeRobotDataset requires at least one source.")
        if viewpoint != "concat_view":
            raise NotImplementedError("UR5SingleEEFLeRobotDataset only supports concat_view.")
        self._source_specs = [_normalize_source(source) for source in sources]
        self._source_tasks: list[dict[int, str]] = []
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="ur5-single-eef",
            viewpoint=viewpoint,
            pose_convention="backward_framewise",
            rotation_format="rot6d",
            action_normalization=action_normalization,
            tolerance_s=max(source.tolerance_s for source in self._source_specs),
            skip_video_loading=skip_video_loading,
            sample_stride=sample_stride,
        )

        for source in self._source_specs:
            meta = LeRobotDatasetMetadata(repo_id="local", root=source.root, revision="local")
            _validate_source(meta, source, fps=self._fps)
            self._register_source(
                root=source.root,
                delta_timestamps=_delta_timestamps(source, fps=self._fps, chunk_length=self._chunk_length),
                tolerance_s=source.tolerance_s,
                download_videos=False,
                video_backend=video_backend,
                dataset_label=source.name or Path(source.root).name,
                prefetched_meta=meta,
            )
            self._source_tasks.append(_task_lookup(meta))

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def source_specs(self) -> tuple[UR5SingleEEFSourceSpec, ...]:
        return tuple(self._source_specs)

    def _build_action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    def _get_dataset(self, ds_idx: int) -> LeRobotDataset:
        dataset = super()._get_dataset(ds_idx)
        if not self._skip_video_loading:
            cameras = set(self._source_specs[ds_idx].camera_features.values())
            dataset.meta.info["features"] = {
                key: value
                for key, value in dataset.meta.info["features"].items()
                if value.get("dtype") != "video" or key in cameras
            }
        return dataset

    def _compose_canvas(self, sample: dict[str, Any], source: UR5SingleEEFSourceSpec) -> torch.Tensor | None:
        if self._skip_video_loading:
            return None
        views = {
            role: resize_view(sample.pop(source.camera_features[role]), source.decode_size_hw) for role in _CAMERA_ROLES
        }
        return concat_three_view_canvas(views["primary"], views["aux_left"], views["aux_right"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, source_index, _, sample = self._fetch_sample(int(idx))
        source = self._source_specs[source_index]
        action = _assemble_action(source, sample, chunk_length=self._chunk_length)
        task_index = int(torch.as_tensor(sample["task_index"]).item())
        try:
            task = self._source_tasks[source_index][task_index]
        except KeyError as error:
            raise ValueError(f"UR5-single EEF sample references missing task index {task_index}.") from error
        captions = [part.strip() for part in task.split("|") if part.strip()]
        return self._build_result(
            mode=mode,
            video=self._compose_canvas(sample, source),
            action=action,
            ai_caption=random.choice(captions or [task]),
            additional_view_description=source.view_description,
        )


def get_action_ur5_single_eef_sft_dataset(
    *,
    sources: Sequence[UR5SingleEEFSourceSpec | Mapping[str, Any]],
    fps: float = 15.0,
    chunk_length: int = 32,
    sample_stride: int = 1,
    split: str = "full",
    split_seed: int = 42,
    split_val_ratio: float = 0.0,
    mode: str = "policy",
    action_normalization: ActionNormalization | None = None,
    viewpoint: str = "concat_view",
    video_backend: str | None = "torchcodec",
    skip_video_loading: bool = False,
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.0,
    action_channel_masking: bool = True,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> Dataset:
    """Build the source-neutral UR5 EEF action SFT dataset."""

    dataset = UR5SingleEEFLeRobotDataset(
        sources=sources,
        fps=fps,
        chunk_length=chunk_length,
        sample_stride=sample_stride,
        split=split,
        split_seed=split_seed,
        split_val_ratio=split_val_ratio,
        mode=mode,
        viewpoint=viewpoint,
        action_normalization=action_normalization,
        video_backend=video_backend,
        skip_video_loading=skip_video_loading,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        action_channel_masking=action_channel_masking,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft


def _manifest_source_configs(manifest: Any) -> list[dict[str, Any]]:
    """Translate manifest dataset descriptions into explicit source specs."""

    model = manifest.model_action
    wire = manifest.wire_action
    conditioning = manifest.conditioning
    if manifest.robot != "ur5" or manifest.domain_name != "ur5-single-eef":
        raise ValueError("UR5-single EEF adapters require robot='ur5' and domain_name='ur5-single-eef'.")
    if model.codec != "eef_delta" or model.representation != "delta" or len(model.layout) != _ACTION_DIM:
        raise ValueError("UR5-single EEF adapters require a canonical 10-D eef_delta model action.")
    if tuple(model.layout) != _EEF_DELTA_LAYOUT:
        raise ValueError(f"UR5-single EEF model action layout must be {_EEF_DELTA_LAYOUT!r}.")
    if model.gripper.index != 9 or model.gripper.semantics != "close_fraction":
        raise ValueError("UR5-single EEF adapters emit close_fraction model gripper actions at index 9.")
    if wire.codec != "eef_absolute" or model.frame != wire.frame or not model.frame:
        raise ValueError("UR5-single EEF model/wire actions require one shared non-empty EEF frame.")
    if (conditioning.state_rows, conditioning.history_rows, conditioning.source) != (0, 0, "none"):
        raise ValueError("UR5-single EEF policies require 0/0 no-state conditioning.")

    source_configs: list[dict[str, Any]] = []
    for source in manifest.datasets:
        if source.condition_source != "none" or source.state_features:
            raise ValueError(f"UR5-single EEF source {source.name!r} must use no state conditioning.")
        if len(source.action_features) != 2:
            raise ValueError(f"UR5-single EEF source {source.name!r} requires [pose_feature, gripper_action_feature].")
        if tuple(source.action_layout) != tuple(model.layout):
            raise ValueError(f"UR5-single EEF source {source.name!r} action layout does not match the model.")
        if set(source.camera_features) != set(_CAMERA_ROLES):
            raise ValueError(f"UR5-single EEF source {source.name!r} must declare all three camera roles.")
        source_frame = source.source_frame
        if not source_frame or source_frame != model.frame or source_frame != wire.frame:
            raise ValueError(
                f"UR5-single EEF source {source.name!r} frame must equal the model/wire frame {model.frame!r}."
            )
        quaternion_order = source.source_quaternion_order
        if quaternion_order not in ("xyzw", "wxyz"):
            raise ValueError(f"UR5-single EEF source {source.name!r} requires an explicit quaternion order.")
        gripper_index = source.source_gripper_index
        if not isinstance(gripper_index, int) or gripper_index < 0:
            raise ValueError(f"UR5-single EEF source {source.name!r} requires a non-negative gripper index.")
        target_offset = source.source_target_offset
        if target_offset != 1:
            raise ValueError(f"UR5-single EEF source {source.name!r} requires source_target_offset=1.")
        source_configs.append(
            {
                "root": source.root,
                "name": source.name,
                "pose_feature": source.action_features[0],
                "quaternion_order": quaternion_order,
                "gripper_action_feature": source.action_features[1],
                "gripper_index": gripper_index,
                "gripper_target_offset": target_offset,
                "eef_frame": source_frame,
                "camera_features": dict(source.camera_features),
                "action_layout": tuple(source.action_layout),
                "gripper_semantics": source.gripper_semantics,
                "view_description": source.view_description,
                "tolerance_s": _DEFAULT_TOLERANCE_S,
                "decode_size_hw": tuple(manifest.observation.view_shape_hw),
            }
        )
    return source_configs


def _bind_ur5_single_eef_action_policy_manifest(manifest: Any) -> list[dict[str, Any]]:
    """Factory hook used by policy_schema to bind manifest-owned sources."""

    return _manifest_source_configs(manifest)


def _validate_ur5_single_eef_action_policy_manifest(dataset_config: Any, manifest: Any) -> None:
    """Reject semantic drift between resolved factory sources and the manifest."""

    expected = [_normalize_source(source) for source in _manifest_source_configs(manifest)]
    raw_sources = (
        dataset_config.get("sources")
        if isinstance(dataset_config, Mapping)
        else getattr(dataset_config, "sources", None)
    )
    if raw_sources is None:
        raise ValueError("Resolved UR5-single EEF dataset config has no sources.")
    actual = [_normalize_source(source) for source in raw_sources]
    if len(actual) != len(expected):
        raise ValueError(f"Manifest describes {len(expected)} EEF source(s), resolved factory has {len(actual)}.")
    field_names = [field.name for field in fields(UR5SingleEEFSourceSpec)]
    differences = []
    for index, (resolved, described) in enumerate(zip(actual, expected, strict=True)):
        for field_name in field_names:
            resolved_value = getattr(resolved, field_name)
            expected_value = getattr(described, field_name)
            if resolved_value != expected_value:
                differences.append(
                    f"sources[{index}].{field_name}: resolved={resolved_value!r}, manifest={expected_value!r}"
                )
    if differences:
        raise ValueError("UR5-single EEF manifest does not match the resolved dataset: " + "; ".join(differences))


setattr(
    get_action_ur5_single_eef_sft_dataset,
    "action_policy_manifest_binder",
    _bind_ur5_single_eef_action_policy_manifest,
)
setattr(
    get_action_ur5_single_eef_sft_dataset,
    "action_policy_manifest_validator",
    _validate_ur5_single_eef_action_policy_manifest,
)


__all__ = [
    "UR5SingleEEFLeRobotDataset",
    "UR5SingleEEFSourceSpec",
    "get_action_ur5_single_eef_sft_dataset",
]
