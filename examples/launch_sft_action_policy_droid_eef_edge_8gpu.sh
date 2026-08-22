#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# One-node, 8-GPU Cosmos3-Edge DROID Fx4 EEF Stage-1 training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOML_FILE="$SCRIPT_DIR/toml/sft_config/action_policy_droid_eef_edge_8gpu.toml"
PFS_ROOT="/mnt/pfs/swy"
DROID_ROOT="$PFS_ROOT/dataset/DROID/Cosmos3-DROID"
FX4_MANIFEST_DIR="$PFS_ROOT/dataset/DROID/manifests/fx4_droid_franka_eef_c32_v2"
FX4_TRAIN_MANIFEST="$FX4_MANIFEST_DIR/manifest_train.jsonl"
ACTION_STATS_PATH="$PFS_ROOT/dataset/DROID/stats/fx4_droid_franka_eef_c32_v2_quantile_rot.json"
EDGE_BASE_PATH="$PFS_ROOT/cosmos3/models/Cosmos3-Edge"
BASE_CHECKPOINT_PATH="$PFS_ROOT/cosmos3/checkpoints/Cosmos3-Edge-DCP"
WAN_VAE_PATH="$PFS_ROOT/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
OUTPUT_ROOT="$PFS_ROOT/cosmos3/runs/cosmos3_action_droid_eef"
PYTHON_BIN="$PFS_ROOT/cosmos3/python/cpython-3.13.12-linux-x86_64-gnu/bin/python3.13"
NPROC_PER_NODE=8
NNODES=1
unset MASTER_ADDR NODE_RANK

export ACTION_STATS_PATH BASE_CHECKPOINT_PATH DROID_ROOT EDGE_BASE_PATH FX4_TRAIN_MANIFEST
export NNODES NPROC_PER_NODE OUTPUT_ROOT PYTHON_BIN WAN_VAE_PATH
export LD_LIBRARY_PATH=""
DATASET_PATH="$DROID_ROOT"

if [[ ! -f "$EDGE_BASE_PATH/config.json" || ! -f "$EDGE_BASE_PATH/model.safetensors.index.json" ]]; then
    echo "ERROR: EDGE_BASE_PATH is not a complete Cosmos3-Edge snapshot: $EDGE_BASE_PATH" >&2
    exit 1
fi

"${PYTHON_BIN:-python}" "$SCRIPT_DIR/../tools/droid/validate_droid_eef_training.py" \
    --toml "$TOML_FILE" --world-size 8 \
    --dataset-root "$DROID_ROOT" --manifest-dir "$FX4_MANIFEST_DIR" \
    --action-stats "$ACTION_STATS_PATH" || exit 1

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$SCRIPT_DIR/_sft_launcher_common.sh"
