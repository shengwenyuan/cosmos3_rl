#!/usr/bin/env bash
# Source-neutral UR5 single-arm joint-position policy post-training.

TOML_FILE="examples/toml/sft_config/action_policy_ur5_single_joint_overfit.toml"
: "${DATASET_PATH:=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robolabsim-joints-147}"
: "${BASE_CHECKPOINT_PATH:=/mlp_vepfs/share/swy/cosmos3-framework/checkpoints/Cosmos3-Nano-dcp}"

export UR5_SINGLE_ROOT="${UR5_SINGLE_ROOT:-$DATASET_PATH}"

read -r -a TAIL_OVERRIDES <<< "${EXTRA_TAIL_OVERRIDES:-}"

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
