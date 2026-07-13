# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_droid_nano`` — Cosmos3-Nano DROID action policy SFT recipe.

Mirrors the vision SFT stack (PackingDataLoader + RankPartitionedDataLoader),
but feeds the DROID action dataset (``joint_pos`` 8D + ``use_state``, raw/
un-normalized) through ``ActionTransformPipeline``, and trains the generation +
action heads from the public ``nvidia/Cosmos3-Nano`` base.

Usage (1 node, 8 GPU)::

    DROID_ROOT=/path/to/Cosmos3-DROID/success \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_droid_repro.toml
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config._action_policy_nano_common import (
    build_action_policy_nano_experiment,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_droid_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()

_dataset = L(get_action_droid_sft_dataset)(
    root="${oc.env:DROID_ROOT}",
    dataset_profile="cosmos3_droid_success_640x360_v1",
    fps=15.0,
    chunk_length=32,
    action_space="joint_pos",
    # Policy-only task mode avoids diluting each per-task loss across the
    # forward-dynamics, inverse-dynamics, and policy objectives.
    mode="policy",
    use_state=True,
    iterable_shuffle=True,  # rank x worker episode-shuffle stream
    episode_shuffle_seed=42,
    # DROID's CPU-side random crop/rescale + ColorJitter augmentation.
    use_image_augmentation=True,
    # Optional keep-range filtering remains disabled in the shipped recipe.
    use_filter_dict=False,
    filter_dict_path=None,
    action_normalization=None,
    viewpoint="concat_view",  # wrist view above two opposing shoulder views
    resolution="480",  # 640x360 source data on the 480p canvas
    max_action_dim="${model.config.max_action_dim}",
    cfg_dropout_rate=0.1,
    tokenizer_config="${model.config.vlm_config.tokenizer}",
    action_channel_masking=True,
    append_viewpoint_info=True,
    append_duration_fps_timestamps=True,
    append_resolution_info=True,
    append_idle_frames=False,
    format_prompt_as_json=True,  # match the DROID action-policy prompt format
)

action_policy_droid_nano = build_action_policy_nano_experiment(
    name="action_policy_droid_nano",
    dataset_name="action_droid",
    dataset_key="droid",
    dataset_node=_dataset,
    callbacks=("basic", "optimization", "job_monitor"),
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_droid_nano",
    node=action_policy_droid_nano,
)
