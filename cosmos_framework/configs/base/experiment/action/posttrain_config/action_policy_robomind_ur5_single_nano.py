# UR5e post-training — local addition, not part of upstream Cosmos3.

"""``action_policy_robomind_ur5_single_nano`` — single-arm UR5e policy SFT (RoboLab-bound).

Post-trains Cosmos3-Nano on **single-arm** RoboMIND UR5e data (7-D `[joint(6), gripper(1)]`), the
policy-DROID analogue evaluated on a single-arm robot. Source: RoboMIND 1.2 ``h5_ur_1rgb`` converted
to LeRobot v3 (the single arm lands in the ``left`` slot). Point ``UR5_SINGLE_ROOT`` at it.

Usage (1 node, 8 GPU)::

    UR5_SINGLE_ROOT=/path/to/robomind_ur5_single_lerobot/success \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config._robomind_ur5_common import (
    build_robomind_ur5_experiment,
)

cs = ConfigStore.instance()

action_policy_robomind_ur5_single_nano = build_robomind_ur5_experiment(
    name="action_policy_robomind_ur5_single_nano",
    which_arm="left",
    root_env="UR5_SINGLE_ROOT",
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robomind_ur5_single_nano",
    node=action_policy_robomind_ur5_single_nano,
)
