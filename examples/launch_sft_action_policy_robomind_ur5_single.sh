#!/usr/bin/env bash
# UR5e post-training — local addition, not part of upstream Cosmos3.

# ============================================================================
# Structured-TOML launch for RoboMIND UR5(e) SINGLE-ARM action-policy SFT on
# Cosmos3-Nano. Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml (selects
# the registered `action_policy_robomind_ur5_single_nano` experiment; res480,
# single-arm joint_pos 7D + use_state). See docs/action_policy_robomind_ur5_posttrain.md.
#
# Env vars (override for your filesystem):
#   DATASET_PATH          RoboMIND UR5e single-arm LeRobot v3 success split (from h5_ur_1rgb)
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#
# Single-node smoke (config/data sanity, a few iters):
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
#                                dataloader_train.max_samples_per_batch=8"
#   bash examples/launch_sft_action_policy_robomind_ur5_single.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml"
: "${DATASET_PATH:=examples/data/lerobot_v30/robomind_ur5_single_lerobot/success}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

# The experiment reads ${oc.env:UR5_SINGLE_ROOT}; bridge the launcher's DATASET_PATH to it.
export UR5_SINGLE_ROOT="${UR5_SINGLE_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$UR5_SINGLE_ROOT/meta/info.json" ]] || { echo "ERROR: missing $UR5_SINGLE_ROOT/meta/info.json (convert RoboMIND 1.2 h5_ur_1rgb to LeRobot v3 — see docs/action_policy_robomind_ur5_posttrain.md)" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
