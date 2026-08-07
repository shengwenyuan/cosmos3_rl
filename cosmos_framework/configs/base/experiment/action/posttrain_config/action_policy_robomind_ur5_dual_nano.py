# UR5e post-training — local addition, not part of upstream Cosmos3.

"""``action_policy_robomind_ur5_dual_nano`` — dual-arm UR5e policy SFT (secondary case).

Post-trains Cosmos3-Nano on **dual-arm** RoboMIND 2.0 UR5e data (14-D
`[L_joint(6), L_grip, R_joint(6), R_grip]`). Not directly runnable on the single-arm RoboLab; kept
as a second embodiment. Source: RoboMIND 2.0 HDF5 converted to LeRobot v3; point ``UR5_DUAL_ROOT`` at it.

Usage (1 node, 8 GPU)::

    UR5_DUAL_ROOT=/path/to/robomind_ur5_dual_lerobot/success \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \\
        --sft-toml examples/toml/sft_config/action_policy_robomind_ur5_dual_repro.toml
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config._robomind_ur5_common import (
    build_robomind_ur5_experiment,
)

cs = ConfigStore.instance()

action_policy_robomind_ur5_dual_nano = build_robomind_ur5_experiment(
    name="action_policy_robomind_ur5_dual_nano",
    which_arm="dual",
    root_env="UR5_DUAL_ROOT",
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robomind_ur5_dual_nano",
    node=action_policy_robomind_ur5_dual_nano,
)
