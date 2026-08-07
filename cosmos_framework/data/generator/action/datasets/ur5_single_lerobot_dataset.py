# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Source-neutral LeRobot adapter for 7-D UR5 joint policies.

Each source declares its storage schema explicitly.  Both supported schemas
produce 33 rows: the conditioning joint state/action at row 0 followed by 32
supervised actions, all laid out as ``[joint(6), gripper(1)]``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.generator.action.datasets.canvas_utils import (
    concat_three_view_canvas,
    resize_view,
    zero_like_view,
)
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    BaseActionLeRobotDataset,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

ConditionSource = Literal["action_t0", "observation_state_t0"]
GripperSemantics = Literal["close_fraction", "open_fraction"]

_ACTION_DIM = 7
_JOINT_DIM = 6
_CAMERA_ROLES = frozenset(("primary", "aux_left", "aux_right"))


@dataclass(frozen=True, slots=True)
class UR5SingleSourceSpec:
    """Explicitly map one LeRobot storage layout into the canonical policy."""

    root: str
    name: str
    condition_source: ConditionSource
    joint_action_feature: str
    gripper_action_feature: str | None
    camera_features: Mapping[str, str]
    action_layout: tuple[str, ...]
    gripper_semantics: GripperSemantics
    view_description: str
    joint_state_feature: str | None = None
    gripper_state_feature: str | None = None
    tolerance_s: float = 2e-4
    decode_size_hw: tuple[int, int] | None = (360, 640)


def _normalize_source(source: UR5SingleSourceSpec | Mapping[str, Any]) -> UR5SingleSourceSpec:
    if isinstance(source, UR5SingleSourceSpec):
        return source
    values = dict(source)
    for key in ("action_layout", "decode_size_hw"):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    if values.get("camera_features") is not None:
        values["camera_features"] = dict(values["camera_features"])
    return UR5SingleSourceSpec(**values)


def _require_feature(features: Mapping[str, Any], key: str, shape: tuple[int, ...]) -> Mapping[str, Any]:
    feature = features.get(key)
    if not isinstance(feature, Mapping):
        raise ValueError(f"Missing required UR5-single feature {key!r}.")
    actual_shape = tuple(feature.get("shape", ()))
    if feature.get("dtype") != "float32" or actual_shape != shape:
        raise ValueError(f"Feature {key!r} must be float32 with shape {shape}, got {dict(feature)!r}.")
    return feature


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
        raise ValueError("UR5-single source metadata contains no usable task text.")
    return result


def _validate_source(meta: LeRobotDatasetMetadata, source: UR5SingleSourceSpec, *, fps: float) -> None:
    roles = set(source.camera_features)
    if "primary" not in roles or not roles <= _CAMERA_ROLES:
        raise ValueError(
            "camera_features must map 'primary' and optional 'aux_left'/'aux_right' roles to feature names"
        )
    if not source.name.strip() or not source.view_description.strip():
        raise ValueError("UR5-single source name and view_description must not be empty.")
    if len(source.action_layout) != _ACTION_DIM:
        raise ValueError("UR5-single sources require an explicit 7-name action_layout.")
    if float(meta.fps) != float(fps):
        raise ValueError(f"Source fps={meta.fps} does not match policy fps={fps}; implicit resampling is forbidden.")

    features = meta.info.get("features", {})
    if source.gripper_action_feature is None:
        feature = _require_feature(features, source.joint_action_feature, (_ACTION_DIM,))
        names = feature.get("names")
        actual_layout = tuple(names.get("motors", ())) if isinstance(names, Mapping) else tuple(names or ())
        if actual_layout and actual_layout != source.action_layout:
            raise ValueError(
                f"UR5 action layout mismatch: metadata has {actual_layout!r}, "
                f"source contract requires {source.action_layout!r}."
            )
    else:
        _require_feature(features, source.joint_action_feature, (_JOINT_DIM,))
        _require_feature(features, source.gripper_action_feature, (1,))

    if source.condition_source == "observation_state_t0":
        if source.joint_state_feature is None or source.gripper_state_feature is None:
            raise ValueError("observation_state_t0 requires explicit joint_state_feature and gripper_state_feature.")
        _require_feature(features, source.joint_state_feature, (_JOINT_DIM,))
        _require_feature(features, source.gripper_state_feature, (1,))
    elif source.joint_state_feature is not None or source.gripper_state_feature is not None:
        raise ValueError("state features are only valid when condition_source='observation_state_t0'.")

    for camera in source.camera_features.values():
        feature = features.get(camera)
        if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
            raise ValueError(f"Camera feature {camera!r} is missing or is not video.")
        camera_fps = feature.get("info", {}).get("video.fps")
        if camera_fps is not None and float(camera_fps) != float(fps):
            raise ValueError(f"Camera {camera!r} fps={camera_fps} does not match policy fps={fps}.")


def _delta_timestamps(source: UR5SingleSourceSpec, *, fps: float, chunk_length: int) -> dict[str, list[float]]:
    start = 0 if source.condition_source == "action_t0" else 1
    action_times = [index / fps for index in range(start, chunk_length + 1)]
    timestamps = {source.joint_action_feature: action_times}
    if source.gripper_action_feature is not None:
        timestamps[source.gripper_action_feature] = action_times
    if source.condition_source == "observation_state_t0":
        assert source.joint_state_feature is not None and source.gripper_state_feature is not None
        timestamps[source.joint_state_feature] = [0.0]
        timestamps[source.gripper_state_feature] = [0.0]
    return timestamps


def _rows(sample: Mapping[str, Any], key: str, rows: int, width: int) -> torch.Tensor:
    value = torch.as_tensor(sample[key]).float()
    # LeRobot squeezes scalar ``shape=[1]`` time-series to ``[T]``. Restore
    # only that metadata-defined scalar channel; every other shape stays strict.
    if width == 1 and tuple(value.shape) == (rows,):
        value = value.unsqueeze(-1)
    expected = (rows, width)
    if tuple(value.shape) != expected:
        raise ValueError(f"UR5 feature {key!r} has sample shape {tuple(value.shape)}, expected {expected}.")
    return value


def _assemble_vectors(
    sample: Mapping[str, Any], joint_feature: str, gripper_feature: str | None, rows: int
) -> torch.Tensor:
    if gripper_feature is None:
        return _rows(sample, joint_feature, rows, _ACTION_DIM)
    return torch.cat(
        (_rows(sample, joint_feature, rows, _JOINT_DIM), _rows(sample, gripper_feature, rows, 1)),
        dim=-1,
    )


def _assemble_action(source: UR5SingleSourceSpec, sample: Mapping[str, Any], *, chunk_length: int) -> torch.Tensor:
    target_rows = chunk_length + (1 if source.condition_source == "action_t0" else 0)
    targets = _assemble_vectors(
        sample,
        source.joint_action_feature,
        source.gripper_action_feature,
        target_rows,
    )
    if source.condition_source == "action_t0":
        action = targets
    else:
        assert source.joint_state_feature is not None and source.gripper_state_feature is not None
        condition = _assemble_vectors(sample, source.joint_state_feature, source.gripper_state_feature, 1)
        action = torch.cat((condition, targets), dim=0)
    if source.gripper_semantics == "open_fraction":
        action = action.clone()
        action[:, -1] = 1.0 - action[:, -1]
    return action


class UR5SingleLeRobotDataset(BaseActionLeRobotDataset):
    """Canonical 7-D UR5 joint policy over one or more explicit sources."""

    def __init__(
        self,
        *,
        sources: Sequence[UR5SingleSourceSpec | Mapping[str, Any]],
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
            raise ValueError("UR5SingleLeRobotDataset requires at least one source.")
        if viewpoint != "concat_view":
            raise NotImplementedError("UR5SingleLeRobotDataset only supports concat_view.")
        self._source_specs = [_normalize_source(source) for source in sources]
        self._source_tasks: list[dict[int, str]] = []
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="ur5-single-joint",
            viewpoint=viewpoint,
            pose_convention="backward_framewise",
            rotation_format=None,
            action_normalization=action_normalization,
            tolerance_s=max(source.tolerance_s for source in self._source_specs),
            skip_video_loading=skip_video_loading,
            sample_stride=sample_stride,
        )

        observation_ts = [index / self._fps for index in range(self._chunk_length + 1)]
        for source in self._source_specs:
            meta = LeRobotDatasetMetadata(repo_id="local", root=source.root, revision="local")
            _validate_source(meta, source, fps=self._fps)
            timestamps = _delta_timestamps(source, fps=self._fps, chunk_length=self._chunk_length)
            timestamps.update({camera: observation_ts for camera in source.camera_features.values()})
            self._register_source(
                root=source.root,
                delta_timestamps=timestamps,
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
    def source_specs(self) -> tuple[UR5SingleSourceSpec, ...]:
        return tuple(self._source_specs)

    def _build_action_spec(self) -> ActionSpec:
        return build_action_spec(Joint(n=_JOINT_DIM, label="joint"), Gripper())

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

    def _compose_canvas(self, sample: dict[str, Any], source: UR5SingleSourceSpec) -> torch.Tensor | None:
        if self._skip_video_loading:
            return None
        views = {
            role: resize_view(sample.pop(camera), source.decode_size_hw)
            for role, camera in source.camera_features.items()
        }
        top = views["primary"]
        left = views.get("aux_left", zero_like_view(top))
        right = views.get("aux_right", zero_like_view(top))
        return concat_three_view_canvas(top, left, right)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, source_index, _, sample = self._fetch_sample(int(idx))
        source = self._source_specs[source_index]
        action = _assemble_action(source, sample, chunk_length=self._chunk_length)
        task_index = int(torch.as_tensor(sample["task_index"]).item())
        try:
            task = self._source_tasks[source_index][task_index]
        except KeyError as error:
            raise ValueError(f"UR5-single sample references missing task index {task_index}.") from error
        captions = [part.strip() for part in task.split("|") if part.strip()]
        return self._build_result(
            mode=mode,
            video=self._compose_canvas(sample, source),
            action=action,
            ai_caption=random.choice(captions or [task]),
            additional_view_description=source.view_description,
        )


def get_action_ur5_single_sft_dataset(
    *,
    sources: Sequence[UR5SingleSourceSpec | Mapping[str, Any]],
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
    """Build the source-neutral UR5 single-arm action SFT dataset."""

    dataset = UR5SingleLeRobotDataset(
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


def _bind_ur5_joint_manifest_sources(manifest: Any) -> list[dict[str, Any]]:
    """Translate the artifact source descriptions into joint reader specs."""

    if manifest.robot != "ur5" or manifest.domain_name != "ur5-single-joint":
        raise ValueError("UR5 joint adapters require robot='ur5' and domain_name='ur5-single-joint'")
    if manifest.model_action.codec != "joint_position" or manifest.model_action_dim != _ACTION_DIM:
        raise ValueError("UR5 joint adapters require a 7-D joint-position model action")
    if manifest.model_action.gripper.index != _ACTION_DIM - 1:
        raise ValueError("UR5 joint adapters require the model gripper at index 6")
    if manifest.model_action.gripper.semantics != "close_fraction":
        raise ValueError("UR5 joint adapters emit canonical close_fraction model actions")

    source_configs: list[dict[str, Any]] = []
    for source in manifest.datasets:
        if source.condition_source == "none":
            raise ValueError("UR5 joint adapters require action_t0 or observation_state_t0 conditioning")
        if len(source.action_features) not in (1, 2):
            raise ValueError("UR5 joint adapters require one packed or two split action_features")
        if len(source.state_features) not in (0, 2):
            raise ValueError("UR5 joint adapters require zero or two split state_features")
        if source.action_layout[:-1] != manifest.model_action.layout[:-1]:
            raise ValueError(f"dataset source {source.name!r} joint order does not match the model action layout")
        if len(source.camera_features) < 3 and manifest.observation.missing_view_policy != "black":
            raise ValueError(f"dataset source {source.name!r} has missing camera roles but policy is not 'black'")
        source_configs.append(
            {
                "root": source.root,
                "name": source.name,
                "condition_source": source.condition_source,
                "joint_action_feature": source.action_features[0],
                "gripper_action_feature": source.action_features[1] if len(source.action_features) == 2 else None,
                "joint_state_feature": source.state_features[0] if source.state_features else None,
                "gripper_state_feature": source.state_features[1] if source.state_features else None,
                "camera_features": dict(source.camera_features),
                "action_layout": list(source.action_layout),
                "gripper_semantics": source.gripper_semantics,
                "view_description": source.view_description,
                "decode_size_hw": list(manifest.observation.view_shape_hw),
            }
        )
    return source_configs


def _source_value(source: Any, key: str) -> Any:
    return source.get(key) if isinstance(source, Mapping) else getattr(source, key)


def _validate_ur5_joint_manifest(dataset_config: Any, manifest: Any) -> None:
    """Ensure the resolved source specs remain identical to their manifest."""

    expected_sources = _bind_ur5_joint_manifest_sources(manifest)
    resolved_sources = _source_value(dataset_config, "sources")
    if len(resolved_sources) != len(expected_sources):
        raise ValueError(
            f"manifest describes {len(expected_sources)} dataset source(s), "
            f"resolved UR5 joint factory has {len(resolved_sources)}"
        )
    for resolved, expected in zip(resolved_sources, expected_sources, strict=True):
        for key, expected_value in expected.items():
            actual = _source_value(resolved, key)
            if key in {"action_layout", "decode_size_hw"}:
                actual = list(actual)
            elif key == "camera_features":
                actual = dict(actual)
            if actual != expected_value:
                raise ValueError(
                    f"resolved UR5 joint source {expected['name']!r} field {key!r}="
                    f"{actual!r}, expected {expected_value!r} from the action-policy manifest"
                )


setattr(get_action_ur5_single_sft_dataset, "action_policy_manifest_binder", _bind_ur5_joint_manifest_sources)
setattr(get_action_ur5_single_sft_dataset, "action_policy_manifest_validator", _validate_ur5_joint_manifest)


__all__ = ["UR5SingleLeRobotDataset", "UR5SingleSourceSpec", "get_action_ur5_single_sft_dataset"]
