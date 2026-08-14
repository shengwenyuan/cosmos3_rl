#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# One-node, 8-GPU Cosmos3-Edge LIBERO-all SFT. LIBERO_ROOT must contain the
# four LIBERO_LeRobot_v3 suite directories. The common launcher validates the
# Edge DCP and Wan2.2 VAE paths before starting torchrun.

TOML_FILE="examples/toml/sft_config/action_policy_libero_all_edge_8gpu.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
: "${NPROC_PER_NODE:=8}"

if [[ "$NPROC_PER_NODE" != "8" || "${NNODES:-1}" != "1" ]]; then
    echo "ERROR: this preset requires one node with NPROC_PER_NODE=8; use the 4-GPU preset for four GPUs." >&2
    exit 1
fi

export LIBERO_ROOT="${LIBERO_ROOT:-}"

EXTRA_DATASET_CHECK='for _s in libero_spatial libero_object libero_goal libero_10; do [[ -f "$LIBERO_ROOT/$_s/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must be the LIBERO_LeRobot_v3 parent dir containing all 4 suites (missing $_s; got: $LIBERO_ROOT)." >&2; exit 1; }; done'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
