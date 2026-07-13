#!/usr/bin/env bash
# PAI DSW entry point for the UR5 joint-position overfit recipe.

set -euo pipefail

FAST_ROOT="${FAST_ROOT:-/mlp_vepfs/share/swy/cosmos3-framework}"
export DATASET_PATH="${DATASET_PATH:-$FAST_ROOT/lerobot/robolabsim-joints-147}"
export JOB_NAME="${JOB_NAME:-robolabsim_ur5_joint_overfit}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$FAST_ROOT/outputs/$JOB_NAME}"
export LAUNCHER="${LAUNCHER:-examples/launch_sft_action_policy_ur5_single_joint.sh}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/_launch_paidsw_ur5_action.sh"
