#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Resume the 64-GPU DROID EEF run after verifying its latest DCP checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_ROOT="/mnt/pfs/swy/cosmos3/runs/cosmos3_action_droid_eef/cosmos3_action_droid_eef/stage1_c32/action_policy_droid_eef_edge_64gpu/checkpoints"
EXPECTED_CHECKPOINT="${RESUME_CHECKPOINT:-iter_000002500}"
LATEST_FILE="$CHECKPOINT_ROOT/latest_checkpoint.txt"

[[ -r "$LATEST_FILE" ]] || { echo "ERROR: missing checkpoint pointer: $LATEST_FILE" >&2; exit 1; }
LATEST_CHECKPOINT="$(<"$LATEST_FILE")"
[[ "$LATEST_CHECKPOINT" == "$EXPECTED_CHECKPOINT" ]] || {
    echo "ERROR: latest checkpoint is $LATEST_CHECKPOINT, expected $EXPECTED_CHECKPOINT" >&2
    exit 1
}
for component in model optim scheduler trainer; do
    [[ -r "$CHECKPOINT_ROOT/$LATEST_CHECKPOINT/$component/.metadata" ]] || {
        echo "ERROR: incomplete $component checkpoint: $CHECKPOINT_ROOT/$LATEST_CHECKPOINT" >&2
        exit 1
    }
done

echo ">>> Resuming from $CHECKPOINT_ROOT/$LATEST_CHECKPOINT"
exec "$SCRIPT_DIR/launch_sft_action_policy_droid_eef_edge_64gpu.sh"
