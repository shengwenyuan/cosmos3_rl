#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# One-node, 4-GPU Cosmos3-Edge LIBERO-all SFT. This uses the same effective
# global batch as the 8-GPU recipe by doubling gradient accumulation.

TOML_FILE="examples/toml/sft_config/action_policy_libero_all_edge_4gpu.toml"
: "${BASE_CHECKPOINT_PATH:=${EDGE_DCP_PATH:-examples/checkpoints/Cosmos3-Edge}}"
: "${EDGE_BASE_PATH:=/mnt/cfs/data/swy/cosmos3/models/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=4}"

if [[ "$NPROC_PER_NODE" != "4" || "${NNODES:-1}" != "1" ]]; then
    echo "ERROR: this preset requires one node with NPROC_PER_NODE=4; use the 8-GPU preset for eight GPUs." >&2
    exit 1
fi

export LIBERO_ROOT="${LIBERO_ROOT:-}"
export EDGE_BASE_PATH

if [[ ! -f "$EDGE_BASE_PATH/config.json" || ! -f "$EDGE_BASE_PATH/model.safetensors.index.json" ]]; then
    echo "ERROR: EDGE_BASE_PATH is not a complete Cosmos3-Edge snapshot: $EDGE_BASE_PATH" >&2
    exit 1
fi

EXTRA_DATASET_CHECK='for _s in libero_spatial libero_object libero_goal libero_10; do [[ -f "$LIBERO_ROOT/$_s/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be the LIBERO_LeRobot_v3 parent dir containing all 4 suites (missing $_s; got: $LIBERO_ROOT)." >&2; exit 1; }; done'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
