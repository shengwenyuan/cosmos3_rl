"""Source-neutral single-arm UR5 joint-position policy post-training."""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config._action_policy_nano_common import (
    build_action_policy_nano_experiment,
)
from cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset import (
    get_action_ur5_single_sft_dataset,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

cs = ConfigStore.instance()

_dataset = L(get_action_ur5_single_sft_dataset)(
    # The structured TOML's [action_policy] section binds these sources and
    # all policy/transform scalars after schema validation. This experiment
    # contains no dataset-specific filenames or gripper assumptions.
    sources=[],
    fps=15.0,
    chunk_length=32,
    sample_stride=8,
    max_action_dim="${model.config.max_action_dim}",
    tokenizer_config="${model.config.vlm_config.tokenizer}",
    resolution="480",
    action_channel_masking=True,
    append_viewpoint_info=True,
    append_duration_fps_timestamps=True,
    append_resolution_info=True,
    append_idle_frames=False,
    format_prompt_as_json=True,
    iterable_shuffle=True,
)

action_policy_ur5_single_joint_nano = build_action_policy_nano_experiment(
    name="action_policy_ur5_single_joint_nano",
    dataset_name="action_action_policy_ur5_single_joint_nano",
    dataset_key="ur5_single",
    dataset_node=_dataset,
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_ur5_single_joint_nano",
    node=action_policy_ur5_single_joint_nano,
)
