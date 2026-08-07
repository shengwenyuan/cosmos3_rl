# UR5e post-training — local addition, not part of upstream Cosmos3.

"""Compatibility wrapper for the legacy RoboMIND UR5 experiments."""

from cosmos_framework.configs.base.experiment.action.posttrain_config._action_policy_nano_common import (
    build_action_policy_nano_experiment,
)
from cosmos_framework.data.generator.action.datasets.robomind_ur5_dataset import get_action_robomind_ur5_sft_dataset
from cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset import (
    get_action_ur5_single_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict


def build_robomind_ur5_experiment(*, name: str, which_arm: str, root_env: str) -> LazyDict:
    """Build the existing split-schema RoboMIND single/dual experiment."""

    if which_arm == "left":
        # Single-arm storage fields come entirely from [action_policy.datasets]
        # in the caller's TOML. The old RoboMIND-named adapter remains only for
        # its genuinely different dual-arm action representation.
        dataset_node = L(get_action_ur5_single_sft_dataset)(
            sources=[],
            fps=15.0,
            chunk_length=32,
            mode="policy",
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            action_normalization=None,
            viewpoint="concat_view",
            resolution="480",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            action_channel_masking=True,
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=False,
        )
    else:
        dataset_node = L(get_action_robomind_ur5_sft_dataset)(
            root="${oc.env:" + root_env + "}",
            fps=None,
            chunk_length=32,
            mode="policy",
            use_state=True,
            which_arm=which_arm,
            gripper_invert=False,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            action_normalization=None,
            viewpoint="concat_view",
            resolution="480",
            canvas_layout="three_view_zero_pad",
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
        )
    return build_action_policy_nano_experiment(
        name=name,
        dataset_name=f"action_{name}",
        dataset_key="robomind_ur5",
        dataset_node=dataset_node,
        requires_action_policy_manifest=which_arm == "left",
    )
