#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/cosmos-framework
bash edge_task/configure_bashrc.sh
source edge_task/env.example.sh
export WANDB_ENTITY="$(sed -n '1{s/\r$//;p;}' /mnt/cfs/data/swy/personal/wandb)"
export WANDB_API_KEY="$(sed -n '2{s/\r$//;p;}' /mnt/cfs/data/swy/personal/wandb)"
export NPROC_PER_NODE=8 PYTHONUNBUFFERED=1
unset NNODES NODE_RANK MASTER_ADDR EXTRA_TAIL_OVERRIDES

[[ -x "$COSMOS_PYTHON" ]] || { echo "ERROR: missing Cosmos Python: $COSMOS_PYTHON" >&2; exit 1; }
GPU_COUNT=$("$COSMOS_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
[[ "$GPU_COUNT" == 8 ]] || {
    echo "ERROR: Cosmos training requires 8 visible GPUs; found $GPU_COUNT" >&2
    exit 1
}

if [[ "${EDGE_LIBERO_MODE:-train}" == smoke ]]; then
    exec env EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 checkpoint.save_iter=999999 checkpoint.save_last_checkpoint=false job.group=smoke job.name=action_policy_libero_all_edge_8gpu_smoke job.wandb_mode=disabled" \
        bash examples/launch_sft_action_policy_libero_all_edge_8gpu.sh
fi
[[ "${EDGE_LIBERO_MODE:-train}" == train ]] || { echo "ERROR: EDGE_LIBERO_MODE must be smoke or train" >&2; exit 1; }

exec bash examples/launch_sft_action_policy_libero_all_edge_8gpu.sh
