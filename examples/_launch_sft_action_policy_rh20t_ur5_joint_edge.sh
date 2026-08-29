#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Shared PFS/runtime contract for the short RH20T UR5 joint launchers.

set -uo pipefail

: "${TOML_FILE:?Set TOML_FILE before sourcing this launcher}"
: "${EXPECTED_GPU_WORLD_SIZE:?Set EXPECTED_GPU_WORLD_SIZE before sourcing this launcher}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PFS_ROOT="${PFS_ROOT:-/mnt/pfs/swy}"
RH20T_TRAIN_ROOT="${RH20T_TRAIN_ROOT:-$PFS_ROOT/dataset/RH20T/lerobot/rh20t_cfg4_ur5_joint_ext2_15hz_v1}"
RH20T_DERIVED_ROOT="${RH20T_DERIVED_ROOT:-$PFS_ROOT/dataset/RH20T/derived/stage2_ur5_joint_v1}"
RH20T_TRAIN_MANIFEST="${RH20T_TRAIN_MANIFEST:-$RH20T_DERIVED_ROOT/manifests/train.jsonl}"
ACTION_STATS_PATH="${ACTION_STATS_PATH:-$RH20T_DERIVED_ROOT/stats/train_quantile_7d.json}"
EDGE_BASE_PATH="${EDGE_BASE_PATH:-$PFS_ROOT/cosmos3/models/Cosmos3-Edge}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$PFS_ROOT/cosmos3/checkpoints/Cosmos3-Edge-DCP}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$PFS_ROOT/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PFS_ROOT/cosmos3/runs/cosmos3_action_rh20t_ur5_joint}"
PYTHON_BIN="${PYTHON_BIN:-$PFS_ROOT/cosmos3/envs/cosmos-framework-cu130-train/bin/python}"
WANDB_CREDENTIALS="${WANDB_CREDENTIALS:-$PFS_ROOT/personal/wandb}"
HF_HOME="${HF_HOME:-$PFS_ROOT/cosmos3/cache/huggingface}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PFS_ROOT/cosmos3/cache/wandb}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PFS_ROOT/cosmos3/cache/wandb-data}"
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if (( EXPECTED_GPU_WORLD_SIZE == NPROC_PER_NODE )); then
    NNODES=1
    unset MASTER_ADDR NODE_RANK
else
    # Baige PyTorchJob injects WORLD_SIZE/RANK per pod. Explicit NNODES/
    # NODE_RANK remain accepted for other schedulers; users submit one command.
    NNODES="${NNODES:-${WORLD_SIZE:-}}"
    NODE_RANK="${NODE_RANK:-${RANK:-}}"
    : "${MASTER_ADDR:?PyTorchJob must inject MASTER_ADDR}"
    : "${NNODES:?PyTorchJob must inject WORLD_SIZE, or set NNODES}"
    : "${NODE_RANK:?PyTorchJob must inject RANK, or set NODE_RANK}"
    (( NNODES * NPROC_PER_NODE == EXPECTED_GPU_WORLD_SIZE )) || {
        echo "ERROR: topology gives $((NNODES * NPROC_PER_NODE)) GPUs, expected $EXPECTED_GPU_WORLD_SIZE" >&2
        exit 1
    }
fi

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 1; }
[[ -r "$WANDB_CREDENTIALS" ]] || { echo "ERROR: W&B credentials not found: $WANDB_CREDENTIALS" >&2; exit 1; }
WANDB_ENTITY="$(sed -n '1{s/\r$//;p;}' "$WANDB_CREDENTIALS")"
WANDB_API_KEY="$(sed -n '2{s/\r$//;p;}' "$WANDB_CREDENTIALS")"
[[ -n "$WANDB_ENTITY" && -n "$WANDB_API_KEY" ]] || { echo "ERROR: invalid W&B credentials file" >&2; exit 1; }
[[ -f "$EDGE_BASE_PATH/config.json" && -f "$EDGE_BASE_PATH/model.safetensors.index.json" ]] || {
    echo "ERROR: incomplete Cosmos3-Edge snapshot: $EDGE_BASE_PATH" >&2
    exit 1
}

export ACTION_STATS_PATH BASE_CHECKPOINT_PATH EDGE_BASE_PATH HF_HOME IMAGINAIRE_OUTPUT_ROOT
export MASTER_ADDR NNODES NODE_RANK NPROC_PER_NODE OUTPUT_ROOT PYTHON_BIN RH20T_TRAIN_ROOT
export WANDB_API_KEY WANDB_CACHE_DIR WANDB_DATA_DIR WANDB_ENTITY WAN_VAE_PATH
export LD_LIBRARY_PATH=""
export PATH="$PFS_ROOT/cosmos3/envs/cosmos-framework-cu130-train/bin:$PATH"
DATASET_PATH="$RH20T_TRAIN_ROOT"
LOG_FILENAME="$(basename "$TOML_FILE" .toml)_rank${NODE_RANK:-0}.log"

"$PYTHON_BIN" "$SCRIPT_DIR/../tools/rh20t/validate_rh20t_ur5_joint_training.py" \
    --toml "$TOML_FILE" --world-size "$EXPECTED_GPU_WORLD_SIZE" \
    --dataset-root "$RH20T_TRAIN_ROOT" --train-manifest "$RH20T_TRAIN_MANIFEST" \
    --action-stats "$ACTION_STATS_PATH" || exit 1

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$SCRIPT_DIR/_sft_launcher_common.sh"

