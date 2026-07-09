# UR5e post-training — local addition, not part of upstream Cosmos3.

"""RoboMIND UR joint-space LeRobot dataset for Cosmos Action policy post-training.

Reads a LeRobot dataset produced by ``tools/convert_robomind_hdf5_to_lerobot.py``.
The primary Case B path is ``which_arm="left"``: single-arm RoboMIND 1.0 UR
(7-D, RoboLab-bound). ``which_arm="dual"`` remains available for dual-arm
RoboMIND UR data (14-D), but is not mixed with Berkeley EEF-space training.

Action layout::

    single (7D)  = [joint(6), gripper(1)]
    dual   (14D) = [L_joint(6), L_gripper(1), R_joint(6), R_gripper(1)]

Actions are absolute joint positions, raw (``action_normalization=None``); with ``use_state=True``
the initial observed state is prepended → ``(chunk + 1, dim)``. The action stream is ``puppet``
(executed) by default; the converter can emit ``master`` (leader) instead. The visual canvas is
auto-detected from the cameras present and rendered into a fixed three-view canvas; missing views are zero-padded.
"""

import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torchvision.transforms as T
from lerobot.datasets.video_utils import decode_video_frames

from cosmos_framework.data.vfm.action.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
    ActionIterableShuffleDataset,
    ActionSFTDataset,
)
from cosmos_framework.data.vfm.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.data.vfm.action.datasets.canvas_utils import concat_three_view_canvas, zero_like_view
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

PoseConvention = Literal["backward_framewise"]
Viewpoint = Literal["concat_view"]
WhichArm = Literal["dual", "left", "right"]
CanvasLayout = Literal["auto", "three_view_zero_pad"]

_UR5E_ARM_DOF = 6  # UR5e is a 6-DoF arm (confirmed: arm_*_position_align is (T, 6)).

# --- LeRobot v3 feature names, written by tools/convert_robomind_hdf5_to_lerobot.py -------------
# Known RoboMIND color camera names. The converter writes the cameras present in each source format.
_IMAGE_FEATURES_ALL: dict[str, str] = {
    "top": "observation.images.camera_top",
    "front": "observation.images.camera_front",
    "left": "observation.images.camera_left",
    "right": "observation.images.camera_right",
    "wrist_left": "observation.images.camera_wrist_left",
    "wrist_right": "observation.images.camera_wrist_right",
}
# Canvas auto-detection preference order (used when canvas_views is None). Whatever cameras the
# converter actually wrote drive the layout: 1 present -> top view plus zero-padded missing views;
# an overview + both wrists -> DROID-style 3-view (overview full-width top, wrists half-res bottom,
# ~540x640 at a 360x640 store size), matching the RoboMIND 2.0 VLM's front+wrists observation set.
_CANVAS_PREFERENCE: tuple[str, ...] = ("front", "top", "wrist_left", "wrist_right", "left", "right")

# Per-arm joint / gripper streams. action.* is the imitation target; observation.state.* is the
# prepended proprioceptive state (both puppet-derived by the converter's default).
_ARM_JOINT_ACTION = {"left": "action.arm_left_joint", "right": "action.arm_right_joint"}
_GRIPPER_ACTION = {"left": "action.gripper_left", "right": "action.gripper_right"}
_ARM_JOINT_STATE = {"left": "observation.state.arm_left_joint", "right": "observation.state.arm_right_joint"}
_GRIPPER_STATE = {"left": "observation.state.gripper_left", "right": "observation.state.gripper_right"}


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
    return 7.0


class RoboMINDUR5Dataset(ActionBaseDataset):
    """RoboMIND UR5(e) dataset. Dual-arm 14-D ``joint_pos`` by default::

        [L_joint(6), L_gripper(1), R_joint(6), R_gripper(1)]

    or single-arm 7-D via ``which_arm="left"`` / ``"right"``::

        [joint(6), gripper(1)]
    """

    def __init__(
        self,
        root: str,
        fps: float | None = None,  # None -> read LeRobot meta/info.json fps, fallback 7.0.
        chunk_length: int = 32,
        mode: str = "policy",
        which_arm: WhichArm = "dual",
        pose_convention: PoseConvention = "backward_framewise",
        tolerance_s: float = 2e-4,
        viewpoint: Viewpoint = "concat_view",
        use_state: bool = True,
        # Gripper is a closure fraction (sample: 0≈open at rest, ~0.5-0.6≈closed grasp). Kept raw.
        # NOTE: the repo's RoboMIND-Franka / DROID datasets apply `1 - g`; if you share a serving
        # controller / action convention with them, set gripper_invert=True. Confirm vs your mapping.
        gripper_invert: bool = False,
        canvas_views: tuple[str, ...] | None = None,  # None -> auto-detect from cameras present.
        canvas_layout: CanvasLayout = "three_view_zero_pad",
        action_normalization: str | None = None,  # joint_pos -> raw, no normalization.
        use_image_augmentation: bool = False,
        sample_stride: int = 1,
    ) -> None:
        if which_arm not in ("dual", "left", "right"):
            raise ValueError(f"which_arm must be 'dual' | 'left' | 'right', got {which_arm!r}.")
        if viewpoint != "concat_view":
            raise NotImplementedError("RoboMINDUR5Dataset only supports concat_view.")
        self._which_arm = which_arm
        self._arms: list[str] = ["left", "right"] if which_arm == "dual" else [which_arm]
        self._use_state = bool(use_state)
        self._gripper_invert = bool(gripper_invert)
        self._canvas_layout = canvas_layout
        self._use_image_augmentation = bool(use_image_augmentation)
        self._image_augmentor: T.Compose | None = None
        # Single- and dual-arm map to distinct embodiment ids (7-D vs 14-D action projection).
        domain_name = "robomind-ur5-dual" if which_arm == "dual" else "robomind-ur5-single"
        super().__init__(
            root=root,
            domain_name=domain_name,
            fps=_resolve_fps(root, fps),
            chunk_length=chunk_length,
            mode=mode,
            pose_convention=pose_convention,
            tolerance_s=tolerance_s,
            viewpoint=viewpoint,
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )

        self._require_split_schema()
        self._action_dim = len(self._arms) * (_UR5E_ARM_DOF + 1)

        # Resolve the canvas to the cameras actually present in the dataset (LeRobot info.json).
        self._canvas_features = self._resolve_canvas(canvas_views)

        # Group the (index-sorted) rows into contiguous per-episode blocks and keep only
        # within-episode chunk windows, so a window never straddles two episodes (mirrors DROID's
        # boundary handling; also backs get_shuffle_blocks for the episode-shuffle stream).
        episode_indices = np.asarray([int(r["episode_index"]) for r in self._rows], dtype=np.int64)
        assert np.all(np.diff(episode_indices) >= 0), "episode_index is not contiguous after sorting by frame index"
        _, ep_starts, ep_counts = np.unique(episode_indices, return_index=True, return_counts=True)
        self._ep_starts = ep_starts.astype(np.int64)
        self._valid_cum = np.cumsum(_strided_window_counts(ep_counts, self._chunk_length, self._sample_stride)).astype(
            np.int64
        )

    @property
    def action_dim(self) -> int:
        return int(self._action_dim)  # 14 (dual) or 7 (single)

    def _action_spec(self) -> ActionSpec:
        components: list = []
        for arm in self._arms:
            prefix = arm if self._which_arm == "dual" else ""
            components.append(Joint(n=_UR5E_ARM_DOF, label="joint", prefix=prefix))
            components.append(Gripper(prefix=prefix))
        return build_action_spec(*components)

    @classmethod
    def _stats_path(cls) -> Path:
        # Unused while action_normalization=None (joint_pos is raw). Present for the abstract
        # contract / a future ee_pose (cartesian) variant, which would need computed stats.
        return Path(__file__).parent / "stats/robomind_ur5_stats.json"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        # Map the flat sample index to a within-episode frame window.
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev) * self._sample_stride
        observation_rows = self._rows[start : start + self._chunk_length + 1]
        episode = self._episodes[int(observation_rows[0]["episode_index"])]

        video = self._load_concat_video(episode, observation_rows)
        raw_action = self._build_joint_action(observation_rows)
        task = self._tasks[int(observation_rows[0]["task_index"])]
        ai_caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])

        return self._build_result(
            mode=mode,
            video=video,
            action=raw_action,
            ai_caption=ai_caption,
            additional_view_description=self._view_description(),
        )

    def _resolve_canvas(self, canvas_views: tuple[str, ...] | None) -> list[str]:
        """Resolve the canvas to LeRobot image-feature names present in this dataset.

        Explicit ``canvas_views`` may use RoboMIND role keys (``front``, ``wrist_left``) or exact
        LeRobot feature names. Without an explicit list, RoboMIND cameras keep the
        DROID-style overview+wrists priority, with missing views zero-padded by the loader.
        """
        available = {k for k in self._info.get("features", {}) if k.startswith("observation.images.")}
        if canvas_views is not None:
            feats = []
            for view in canvas_views:
                if view in _IMAGE_FEATURES_ALL:
                    feats.append(_IMAGE_FEATURES_ALL[view])
                elif view.startswith("observation.images."):
                    feats.append(view)
                else:
                    feats.append(f"observation.images.{view}")
            missing = [f for f in feats if f not in available]
            if missing:
                raise ValueError(
                    f"canvas_views requests cameras not in the dataset: {missing}. Have {sorted(available)}."
                )
            return feats

        present = [r for r in _CANVAS_PREFERENCE if _IMAGE_FEATURES_ALL[r] in available]
        if "wrist_left" in present and "wrist_right" in present and any(r in present for r in ("front", "top")):
            overview = next(r for r in ("front", "top") if r in present)
            return [_IMAGE_FEATURES_ALL[r] for r in (overview, "wrist_left", "wrist_right")]
        if present:
            return [_IMAGE_FEATURES_ALL[r] for r in present[:3]]

        raise ValueError(
            "No RoboMIND UR camera features were found. Expected one or more of "
            f"{sorted(_IMAGE_FEATURES_ALL.values())}; have {sorted(available)}."
        )

    def _require_split_schema(self) -> None:
        features = self._info.get("features", {})
        split_required = []
        for arm in self._arms:
            split_required.extend(
                [
                    _ARM_JOINT_ACTION[arm],
                    _GRIPPER_ACTION[arm],
                    _ARM_JOINT_STATE[arm],
                    _GRIPPER_STATE[arm],
                ]
            )
        if all(key in features for key in split_required):
            return
        raise ValueError(
            "Unsupported RoboMIND UR joint-space schema. Expected split fields "
            f"{split_required}. Generic LeRobot `action` datasets such as Berkeley AUTOLab UR5 "
            "must use the Berkeley EEF dataset adapter instead; "
            f"available features: {sorted(features)}"
        )

    def _view_description(self) -> str:
        arm_txt = "dual-arm" if self._which_arm == "dual" else "single-arm"
        n = len(self._canvas_features)
        if n == 3:
            return (
                f"The top row is an overview of the {arm_txt} UR5e workspace. "
                "The bottom row shows two wrist/side camera views side by side."
            )
        if n == 1:
            return f"A view of the {arm_txt} UR5e workspace."
        return f"Concatenated camera views of the {arm_txt} UR5e workspace."

    def _build_joint_action(self, observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        """Absolute joint-position action over the chunk. The window is ``chunk + 1`` frames:
        ``row[0]`` is the initial observed state (prepended when ``use_state``) and ``rows[1:]`` are
        the ``chunk`` commanded actions. Per selected arm: ``[joint(6), gripper(1)]``. No
        normalization (raw joint values)."""
        action_rows = observation_rows[1:]
        per_arm = []
        for arm in self._arms:
            joints = np.asarray([r[_ARM_JOINT_ACTION[arm]] for r in action_rows], dtype=np.float32)  # [chunk, 6]
            gripper = np.asarray([r[_GRIPPER_ACTION[arm]] for r in action_rows], dtype=np.float32).reshape(-1, 1)
            if self._gripper_invert:
                gripper = 1.0 - gripper
            per_arm.append(np.concatenate([joints, gripper], axis=-1))  # [chunk, 7]
        action = np.concatenate(per_arm, axis=-1)  # [chunk, 7*n_arms]

        if self._use_state:
            init = observation_rows[0]
            init_parts = []
            for arm in self._arms:
                init_joint = np.asarray(init[_ARM_JOINT_STATE[arm]], dtype=np.float32)  # [6]
                # Gripper feature is stored as shape (1,); reshape before scalar-izing (np>=1.25 errors on float(1d)).
                init_gripper = np.asarray(init[_GRIPPER_STATE[arm]], dtype=np.float32).reshape(-1)[:1]  # [1]
                if self._gripper_invert:
                    init_gripper = 1.0 - init_gripper
                init_parts.append(np.concatenate([init_joint, init_gripper]))  # [7]
            initial_state = np.concatenate(init_parts)[None, :]  # [1, 7*n_arms]
            action = np.concatenate([initial_state, action], axis=0)  # [chunk + 1, dim]
        return torch.from_numpy(action).float()

    def _load_concat_video(
        self,
        episode: dict[str, Any],
        observation_rows: list[dict[str, Any]],
    ) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]
        frames = [
            decode_video_frames(
                self._video_path(episode, feat),
                [float(episode.get(f"videos/{feat}/from_timestamp", 0.0)) + ts for ts in timestamps],
                self._tolerance_s,
            )
            for feat in self._canvas_features
        ]

        if self._use_image_augmentation and frames:
            # Match DROID: random crop+rescale and color jitter before concat, with
            # one sampled transform shared across every frame and real camera view.
            shapes = {(int(frame.shape[-2]), int(frame.shape[-1])) for frame in frames}
            if len(shapes) != 1:
                raise ValueError(
                    "use_image_augmentation=True requires all real UR5 camera views to share HxW; "
                    f"got {sorted(shapes)}."
                )
            if self._image_augmentor is None:
                _, _, h, w = frames[0].shape
                self._image_augmentor = T.Compose(
                    [
                        T.RandomCrop((int(h * 0.95), int(w * 0.95))),
                        T.Resize((h, w), antialias=True),
                        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                    ]
                )
            counts = [frame.shape[0] for frame in frames]
            augmented = self._image_augmentor(torch.cat(frames, dim=0))
            frames = list(torch.split(augmented, counts, dim=0))

        if self._canvas_layout == "three_view_zero_pad":
            if not frames:
                raise ValueError("No frames decoded for UR5 canvas.")
            top = frames[0]
            left = frames[1] if len(frames) > 1 else zero_like_view(top)
            right = frames[2] if len(frames) > 2 else zero_like_view(top)
            return concat_three_view_canvas(top, left, right)

        if len(frames) == 1:
            return frames[0]
        if len(frames) == 2:
            return torch.cat(frames, dim=-1)  # side by side
        if len(frames) == 3:
            # DROID-style: overview on top (full res), the two wrists half-res side-by-side below.
            return concat_three_view_canvas(frames[0], frames[1], frames[2])
        if len(frames) == 4:
            top = torch.cat([frames[0], frames[1]], dim=-1)
            bottom = torch.cat([frames[2], frames[3]], dim=-1)
            return torch.cat([top, bottom], dim=-2)
        raise NotImplementedError(f"No canvas layout for {len(frames)} views; add one or reduce canvas_views.")

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode flat-index blocks ``(start, length)``. ``ActionIterableShuffleDataset``
        shuffles the ORDER of these blocks and shards them disjointly across ranks, while keeping
        windows *within* a block sequential (decorrelates batches without random-access I/O)."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks


def get_action_robomind_ur5_sft_dataset(
    *,
    root: str,
    fps: float | None = None,  # None -> read LeRobot meta/info.json fps, fallback 7.0.
    chunk_length: int = 32,
    mode: str = "policy",
    which_arm: WhichArm = "dual",
    use_state: bool = True,
    gripper_invert: bool = False,
    canvas_views: tuple[str, ...] | None = None,
    canvas_layout: CanvasLayout = "three_view_zero_pad",
    action_normalization: str | None = None,
    use_image_augmentation: bool = False,
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
    """Build the RoboMIND UR5(e) action SFT dataset: dual-arm ``joint_pos`` (14D, or 7D single-arm)
    + ``use_state`` (raw / un-normalized), concat_view. Mirrors ``get_action_droid_sft_dataset``."""
    dataset = RoboMINDUR5Dataset(
        root=root,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        which_arm=which_arm,
        viewpoint=viewpoint,
        use_state=use_state,
        gripper_invert=gripper_invert,
        canvas_views=canvas_views,
        canvas_layout=canvas_layout,
        action_normalization=action_normalization,
        use_image_augmentation=use_image_augmentation,
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
