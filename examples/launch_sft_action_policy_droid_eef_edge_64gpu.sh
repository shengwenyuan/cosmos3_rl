#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Eight-node, 64-GPU Cosmos3-Edge DROID Fx4 EEF Stage-1 training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOML_FILE="$SCRIPT_DIR/toml/sft_config/action_policy_droid_eef_edge_64gpu.toml"
PFS_ROOT="/mnt/pfs/swy"
DROID_ROOT="$PFS_ROOT/dataset/DROID/Cosmos3-DROID/success"
FX4_MANIFEST_DIR="$PFS_ROOT/dataset/DROID/manifests/fx4_droid_franka_eef_c32_v2"
FX4_TRAIN_MANIFEST="$FX4_MANIFEST_DIR/manifest_train.jsonl"
ACTION_STATS_PATH="$PFS_ROOT/dataset/DROID/stats/fx4_droid_franka_eef_c32_v2_quantile_rot.json"
EDGE_BASE_PATH="$PFS_ROOT/cosmos3/models/Cosmos3-Edge"
BASE_CHECKPOINT_PATH="$PFS_ROOT/cosmos3/checkpoints/Cosmos3-Edge-DCP"
WAN_VAE_PATH="$PFS_ROOT/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
OUTPUT_ROOT="$PFS_ROOT/cosmos3/runs/cosmos3_action_droid_eef"
PYTHON_BIN="$PFS_ROOT/cosmos3/envs/cosmos-framework-cu130-train/bin/python"
HF_HOME="$PFS_ROOT/cosmos3/cache/huggingface"
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
WANDB_CACHE_DIR="$PFS_ROOT/cosmos3/cache/wandb"
WANDB_DATA_DIR="$PFS_ROOT/cosmos3/cache/wandb-data"
WANDB_CREDENTIALS="$PFS_ROOT/personal/wandb"
NPROC_PER_NODE=8
NNODES=8

: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host}"
: "${NODE_RANK:?Set NODE_RANK to 0 through 7}"
[[ "$NODE_RANK" =~ ^[0-7]$ ]] || { echo "ERROR: NODE_RANK must be 0 through 7." >&2; exit 1; }
[[ -r "$WANDB_CREDENTIALS" ]] || { echo "ERROR: W&B credentials not found: $WANDB_CREDENTIALS" >&2; exit 1; }
WANDB_ENTITY="$(sed -n '1{s/\r$//;p;}' "$WANDB_CREDENTIALS")"
WANDB_API_KEY="$(sed -n '2{s/\r$//;p;}' "$WANDB_CREDENTIALS")"
[[ -n "$WANDB_ENTITY" && -n "$WANDB_API_KEY" ]] || { echo "ERROR: invalid W&B credentials file" >&2; exit 1; }

export ACTION_STATS_PATH BASE_CHECKPOINT_PATH DROID_ROOT EDGE_BASE_PATH FX4_TRAIN_MANIFEST
export HF_HOME IMAGINAIRE_OUTPUT_ROOT MASTER_ADDR NNODES NODE_RANK NPROC_PER_NODE
export OUTPUT_ROOT PYTHON_BIN WANDB_API_KEY WANDB_CACHE_DIR WANDB_DATA_DIR WANDB_ENTITY WAN_VAE_PATH
export LD_LIBRARY_PATH=""
export PATH="$PFS_ROOT/cosmos3/envs/cosmos-framework-cu130-train/bin:$PATH"
DATASET_PATH="$DROID_ROOT"
LOG_FILENAME="action_policy_droid_eef_edge_64gpu_rank${NODE_RANK}.log"

if [[ ! -f "$EDGE_BASE_PATH/config.json" || ! -f "$EDGE_BASE_PATH/model.safetensors.index.json" ]]; then
    echo "ERROR: EDGE_BASE_PATH is not a complete Cosmos3-Edge snapshot: $EDGE_BASE_PATH" >&2
    exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/../tools/droid/validate_droid_eef_training.py" \
    --toml "$TOML_FILE" --world-size 64 \
    --dataset-root "$DROID_ROOT" --manifest-dir "$FX4_MANIFEST_DIR" \
    --action-stats "$ACTION_STATS_PATH" || exit 1

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$SCRIPT_DIR/_sft_launcher_common.sh"
