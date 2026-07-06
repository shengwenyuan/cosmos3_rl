# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Berkeley AUTOLab UR5 LeRobot dataset for EEF-space action post-training.

Case A is intentionally separate from the RoboMIND UR joint-space reader. The
Berkeley LeRobot conversion exposes EEF absolute pose state and a generic 7D
EEF command vector. This adapter uses the state pose trajectory to build the
shared Cosmos SE(3) action representation and keeps the native command gripper
channel as the scalar target::

    [se3_delta_translation(3), se3_delta_rot6d(6), gripper(1)]

"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.datasets.canvas_utils import concat_three_view_canvas, zero_like_view
from cosmos_framework.data.vfm.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]

_ACTION_FEATURE = "action"
_STATE_FEATURE = "observation.state"
_EXTERNAL_VIEW = "observation.images.image"
_WRIST_VIEW = "observation.images.hand_image"
_DEPTH_VIEW = "observation.images.image_with_depth"
_IMAGE_PREFERENCE: tuple[str, ...] = (_WRIST_VIEW, _EXTERNAL_VIEW, _DEPTH_VIEW)
_NATIVE_ACTION_DIM = 7
_STATE_DIM = 8
_EEF_ACTION_DIM = 10
_INVALID_FRAME_RE = re.compile(r"Invalid frame index=\d+.*must be less than (\d+)")


def _strided_window_counts(ep_counts: np.ndarray, chunk_length: int, sample_stride: int) -> np.ndarray:
    valid_starts = np.maximum(0, ep_counts.astype(np.int64) - int(chunk_length))
    return ((valid_starts + int(sample_stride) - 1) // int(sample_stride)).astype(np.int64)


def _resolve_fps(root: str, fps: float | None) -> float:
    if fps is not None:
        return float(fps)
    info_path = Path(root) / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        if info.get("fps") is not None:
            return float(info["fps"])
    return 5.0


def _feature_width(info: dict[str, Any], key: str) -> int | None:
    shape = info.get("features", {}).get(key, {}).get("shape")
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    return None


def _row_vector(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float32).reshape(-1)


class BerkeleyUR5EEFDataset(ActionBaseDataset):
    """Berkeley AUTOLab UR5 EEF-space dataset.

    The fixed canvas follows the DROID concat-view layout: wrist camera on top,
    the external Berkeley camera on the bottom-left, and a black zero-padded
    bottom-right view when the third view is absent.
    """

    def __init__(
        self,
        root: str,
        fps: float | None = None,
        chunk_length: int = 32,
        mode: str = "policy",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        canvas_views: tuple[str, ...] | None = None,
        gripper_invert: bool = False,
        action_normalization: str | None = None,
        sample_stride: int = 1,
    ) -> None:
        if viewpoint != "concat_view":
            raise NotImplementedError("BerkeleyUR5EEFDataset only supports concat_view.")
        self._gripper_invert = bool(gripper_invert)
        super().__init__(
            root=root,
            domain_name="berkeley-ur5-eef",
            fps=_resolve_fps(root, fps),
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )

        width = _feature_width(self._info, _ACTION_FEATURE)
        if width != _NATIVE_ACTION_DIM:
            raise ValueError(
                f"Berkeley UR5 EEF adapter expects `{_ACTION_FEATURE}` width {_NATIVE_ACTION_DIM}, got {width}. "
                "If this is a joint-space dataset, use the RoboMIND UR joint adapter instead."
            )
        state_width = _feature_width(self._info, _STATE_FEATURE)
        if state_width != _STATE_DIM:
            raise ValueError(
                f"Berkeley UR5 EEF adapter expects `{_STATE_FEATURE}` width {_STATE_DIM}, got {state_width}. "
                "The state must be [x, y, z, qx, qy, qz, qw, gripper] to build standard SE(3) deltas."
            )
        self._canvas_features = self._resolve_canvas(canvas_views)

        episode_indices = np.asarray([int(r["episode_index"]) for r in self._rows], dtype=np.int64)
        assert np.all(np.diff(episode_indices) >= 0), "episode_index is not contiguous after sorting by frame index"
        _, ep_starts, ep_counts = np.unique(episode_indices, return_index=True, return_counts=True)
        self._ep_starts = ep_starts.astype(np.int64)
        self._valid_cum = np.cumsum(_strided_window_counts(ep_counts, self._chunk_length, self._sample_stride)).astype(
            np.int64
        )

    @property
    def action_dim(self) -> int:
        return _EEF_ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"), Gripper())

    @classmethod
    def _stats_path(cls) -> Path:
        return Path(__file__).parent / "stats/berkeley_ur5_eef_stats.json"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev) * self._sample_stride
        observation_rows = self._rows[start : start + self._chunk_length + 1]
        episode = self._episodes[int(observation_rows[0]["episode_index"])]

        video = self._load_concat_video(episode, observation_rows)
        raw_action = self._build_eef_action(observation_rows, observation_rows[: self._chunk_length])
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            additional_view_description=(
                "The top row is the wrist camera. "
                "The bottom-left row is the external Berkeley UR5 camera; "
                "the bottom-right row is a zero-padded missing view."
            ),
        )

    def _resolve_canvas(self, canvas_views: tuple[str, ...] | None) -> list[str]:
        available = {k for k in self._info.get("features", {}) if k.startswith("observation.images.")}
        if canvas_views is not None:
            feats = [v if v.startswith("observation.images.") else f"observation.images.{v}" for v in canvas_views]
        else:
            feats = [f for f in (_WRIST_VIEW, _EXTERNAL_VIEW) if f in available]
            if not feats:
                feats = [f for f in _IMAGE_PREFERENCE if f in available]
            if _EXTERNAL_VIEW in feats and _DEPTH_VIEW in feats:
                feats = [f for f in feats if f != _DEPTH_VIEW]
        missing = [f for f in feats if f not in available]
        if missing:
            raise ValueError(f"canvas_views requests cameras not in the dataset: {missing}. Have {sorted(available)}.")
        if not feats:
            raise ValueError(f"No Berkeley UR5 observation.images.* cameras found; have {sorted(available)}.")
        return feats[:2]

    def _build_eef_action(
        self,
        observation_rows: list[dict[str, Any]],
        action_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        state = np.asarray([_row_vector(row, _STATE_FEATURE) for row in observation_rows], dtype=np.float32)
        if state.shape[-1] != _STATE_DIM:
            raise ValueError(f"Expected Berkeley state width {_STATE_DIM}, got {state.shape[-1]}.")
        if len(observation_rows) != len(action_rows) + 1:
            raise ValueError(
                f"Expected one more observation row than action rows, got {len(observation_rows)} and {len(action_rows)}."
            )

        poses_abs = build_abs_pose_from_components(state[:, :3], state[:, 3:7], "quat_xyzw")
        poses_rel = pose_abs_to_rel(
            poses_abs,
            rotation_format="rot6d",
            pose_convention=self._pose_convention,
        )
        native = np.asarray([_row_vector(row, _ACTION_FEATURE) for row in action_rows], dtype=np.float32)
        if native.shape[-1] != _NATIVE_ACTION_DIM:
            raise ValueError(f"Expected Berkeley action width {_NATIVE_ACTION_DIM}, got {native.shape[-1]}.")
        if poses_rel.shape[0] != native.shape[0]:
            raise ValueError(f"Pose delta/action length mismatch: {poses_rel.shape[0]} vs {native.shape[0]}.")
        gripper = native[:, 6:7]
        if self._gripper_invert:
            gripper = 1.0 - gripper
        action = np.concatenate([poses_rel, gripper], axis=-1)
        return torch.from_numpy(action).float()

    def _load_concat_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames = [
            self._decode_video_frames_safe(
                self._video_path(episode, feat),
                [float(episode.get(f"videos/{feat}/from_timestamp", 0.0)) + ts for ts in timestamps],
            )
            for feat in self._canvas_features
        ]
        top = frames[0]
        left = frames[1] if len(frames) > 1 else zero_like_view(top)
        right = zero_like_view(top)
        return concat_three_view_canvas(top, left, right)

    def _decode_video_frames_safe(self, video_path: Path, timestamps: list[float]) -> torch.Tensor:
        try:
            return decode_video_frames(video_path, timestamps, self._tolerance_s)
        except IndexError as exc:
            match = _INVALID_FRAME_RE.search(str(exc))
            if match is None:
                raise
            num_frames = int(match.group(1))
            if num_frames <= 0:
                raise
            max_timestamp = (num_frames - 1) / self._fps
            clamped = [min(float(ts), max_timestamp) for ts in timestamps]
            if all(abs(float(a) - float(b)) < 1e-12 for a, b in zip(clamped, timestamps, strict=True)):
                raise
            return decode_video_frames(video_path, clamped, self._tolerance_s)

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks


def get_action_berkeley_ur5_eef_sft_dataset(
    *,
    root: str,
    fps: float | None = None,
    chunk_length: int = 32,
    mode: str = "policy",
    gripper_invert: bool = False,
    canvas_views: tuple[str, ...] | None = None,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    append_idle_frames: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
) -> ActionSFTDataset | ActionIterableShuffleDataset:
    """Build the Berkeley AUTOLab UR5 EEF-space action SFT dataset."""
    dataset = BerkeleyUR5EEFDataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        viewpoint=viewpoint,
        gripper_invert=gripper_invert,
        canvas_views=canvas_views,
        action_normalization=action_normalization,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
        append_idle_frames=append_idle_frames,
    )
    sft = ActionSFTDataset(dataset, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
