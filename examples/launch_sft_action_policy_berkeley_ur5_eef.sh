#!/usr/bin/env bash
# UR5 post-training - local addition, not part of upstream Cosmos3.

# ============================================================================
# Structured-TOML launch for Berkeley AUTOLab UR5 EEF-space action-policy SFT on
# Cosmos3-Nano. This selects the registered `action_policy_berkeley_ur5_eef_nano`
# experiment and predicts 10D EEF delta actions, not RoboMIND joint actions.
#
# Env vars (override for your filesystem):
#   DATASET_PATH          Berkeley AUTOLab UR5 LeRobot dataset
#   BASE_CHECKPOINT_PATH  DCP of nvidia/Cosmos3-Nano (convert_model_to_dcp; see docs)
#   WAN_VAE_PATH          Wan2.2 VAE .pth (Wan-AI/Wan2.2-TI2V-5B)
#   NPROC_PER_NODE        torchrun --nproc_per_node (default 8)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#
# Smoke command once checkpoints are in place:
#   export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
#                                dataloader_train.max_samples_per_batch=8"
#   bash examples/launch_sft_action_policy_berkeley_ur5_eef.sh
# ============================================================================

TOML_FILE="examples/toml/sft_config/action_policy_berkeley_ur5_eef_repro.toml"
: "${DATASET_PATH:=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/berkeley_autolab_ur5}"
: "${BASE_CHECKPOINT_PATH:=/mlp_vepfs/share/swy/cosmos3-framework/checkpoints/Cosmos3-Nano-dcp}"

# The experiment reads ${oc.env:BERKELEY_UR5_ROOT}; bridge the launcher's DATASET_PATH to it.
export BERKELEY_UR5_ROOT="${BERKELEY_UR5_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$BERKELEY_UR5_ROOT/meta/info.json" ]] || { echo "ERROR: missing $BERKELEY_UR5_ROOT/meta/info.json (expected Berkeley AUTOLab UR5 LeRobot dataset)" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
