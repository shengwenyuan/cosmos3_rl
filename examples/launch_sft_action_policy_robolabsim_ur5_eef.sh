#!/usr/bin/env bash
# UR5 post-training - local addition, not part of upstream Cosmos3.

# Structured-TOML launch for RoboLabSim UR5 EEF-space action-policy SFT.

TOML_FILE="examples/toml/sft_config/action_policy_robolabsim_ur5_eef_repro.toml"
: "${DATASET_PATH:=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robolabsim-147}"
: "${BASE_CHECKPOINT_PATH:=/mlp_vepfs/share/swy/cosmos3-framework/checkpoints/Cosmos3-Nano-dcp}"

# The experiment reads ${oc.env:ROBOLABSIM_UR5_ROOT}; bridge the launcher's DATASET_PATH to it.
export ROBOLABSIM_UR5_ROOT="${ROBOLABSIM_UR5_ROOT:-$DATASET_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$ROBOLABSIM_UR5_ROOT/meta/info.json" ]] || { echo "ERROR: missing $ROBOLABSIM_UR5_ROOT/meta/info.json (expected RoboLabSim UR5 LeRobot dataset)" >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
