#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Continue the 64-GPU DROID EEF baseline from iter 3500 to 7000 with a low-LR restart.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="/mnt/pfs/swy/cosmos3/runs/cosmos3_action_droid_eef/cosmos3_action_droid_eef/stage1_c32/action_policy_droid_eef_edge_64gpu/checkpoints/iter_000003500"

for component in model optim trainer; do
    [[ -r "$CHECKPOINT_PATH/$component/.metadata" ]] || {
        echo "ERROR: incomplete $component checkpoint: $CHECKPOINT_PATH" >&2
        exit 1
    }
done

export EXTRA_TAIL_OVERRIDES="\
job.name=action_policy_droid_eef_edge_64gpu_continue_3500_to_7000 \
checkpoint.load_path=$CHECKPOINT_PATH \
checkpoint.load_training_state=true \
checkpoint.keys_not_to_resume=[scheduler,dataloader] \
checkpoint.keys_to_skip_loading=[] \
optimizer.lr=2.0e-6 \
scheduler.cycle_lengths=[3500] \
scheduler.warm_up_steps=[100] \
trainer.max_iter=7000 \
dataloader_train.dataloader.datasets.droid_eef.dataset.episode_shuffle_seed=43"

exec "$SCRIPT_DIR/launch_sft_action_policy_droid_eef_edge_64gpu.sh"
