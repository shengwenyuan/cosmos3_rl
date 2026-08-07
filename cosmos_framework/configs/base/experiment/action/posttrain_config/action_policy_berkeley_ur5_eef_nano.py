# UR5 post-training - local addition, not part of upstream Cosmos3.

"""``action_policy_berkeley_ur5_eef_nano`` - Berkeley AUTOLab UR5 EEF-space policy SFT.

Post-trains Cosmos3-Nano on the Berkeley AUTOLab UR5 LeRobot dataset as a 10D
EEF delta policy: ``[translation_delta(3), rot6d_delta(6), gripper(1)]``. This
is intentionally separate from the RoboMIND UR joint-space recipe.

Usage (1 node, 8 GPU)::

    BERKELEY_UR5_ROOT=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/berkeley_autolab_ur5 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_berkeley_ur5_eef_repro.toml
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config._action_policy_nano_common import (
    build_action_policy_nano_experiment,
)
from cosmos_framework.data.generator.action.datasets.berkeley_ur5_eef_dataset import (
    get_action_berkeley_ur5_eef_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()

_dataset = L(get_action_berkeley_ur5_eef_sft_dataset)(
    root="${oc.env:BERKELEY_UR5_ROOT}",
    fps=5.0,
    chunk_length=32,
    mode="policy",
    gripper_invert=False,
    canvas_views=("observation.images.hand_image", "observation.images.image"),
    decode_size_hw=(360, 640),
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
    format_prompt_as_json=False,
)

action_policy_berkeley_ur5_eef_nano = build_action_policy_nano_experiment(
    name="action_policy_berkeley_ur5_eef_nano",
    dataset_name="action_berkeley_ur5_eef",
    dataset_key="berkeley_ur5_eef",
    dataset_node=_dataset,
    base_lr=1.0e-04,
    action_head_lr_multiplier=2.0,
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_berkeley_ur5_eef_nano",
    node=action_policy_berkeley_ur5_eef_nano,
)
