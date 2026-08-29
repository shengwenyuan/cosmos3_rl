#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Resume the RH20T cfg4 64-GPU run from its latest complete DCP checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/pfs/swy/cosmos3/runs/cosmos3_action_rh20t_ur5_joint/cosmos3_action_rh20t_ur5_joint/stage2_c32/rh20t_cfg4_ur5_joint_edge_64gpu_10k/checkpoints}"
LATEST_FILE="$CHECKPOINT_ROOT/latest_checkpoint.txt"

[[ -r "$LATEST_FILE" ]] || { echo "ERROR: missing checkpoint pointer: $LATEST_FILE" >&2; exit 1; }
LATEST_CHECKPOINT="$(<"$LATEST_FILE")"
for component in model optim scheduler trainer; do
    [[ -r "$CHECKPOINT_ROOT/$LATEST_CHECKPOINT/$component/.metadata" ]] || {
        echo "ERROR: incomplete $component checkpoint: $CHECKPOINT_ROOT/$LATEST_CHECKPOINT" >&2
        exit 1
    }
done

echo ">>> Resuming from $CHECKPOINT_ROOT/$LATEST_CHECKPOINT"
exec "$SCRIPT_DIR/launch_sft_action_policy_rh20t_ur5_joint_edge_64gpu.sh"
