#!/usr/bin/env bash
# Copy into the task platform's private environment configuration.
# Never add HF_TOKEN or other secrets to this file.

export COSMOS_REPO_ROOT=/root/workspace/cosmos-framework
export COSMOS_WRITABLE_ROOT=/mnt/cfs/data/swy/cosmos3
export LIBERO_WRITABLE_ROOT=/mnt/cfs/data/swy/libero

export UV_CACHE_DIR="$COSMOS_WRITABLE_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$COSMOS_WRITABLE_ROOT/python"
export UV_PROJECT_ENVIRONMENT="$COSMOS_WRITABLE_ROOT/envs/cosmos-framework-cu130-train"
export UV_LINK_MODE=hardlink
export HF_HOME="$COSMOS_WRITABLE_ROOT/cache/huggingface"
export HF_ENDPOINT=https://huggingface.co
export IMAGINAIRE_OUTPUT_ROOT="$COSMOS_WRITABLE_ROOT/runs"
export OUTPUT_ROOT="$IMAGINAIRE_OUTPUT_ROOT"
export WANDB_PROJECT=cosmos3_action_libero
export WANDB_CACHE_DIR="$COSMOS_WRITABLE_ROOT/cache/wandb"
export WANDB_DATA_DIR="$COSMOS_WRITABLE_ROOT/cache/wandb-data"

export EDGE_BASE_PATH="$COSMOS_WRITABLE_ROOT/models/Cosmos3-Edge"
export EDGE_DCP_PATH="$COSMOS_WRITABLE_ROOT/checkpoints/Cosmos3-Edge-DCP"
export BASE_CHECKPOINT_PATH="$EDGE_DCP_PATH"
export LIBERO_ROOT="$LIBERO_WRITABLE_ROOT/LIBERO_LeRobot_v3"
export LIBERO_SOURCE_ROOT="$LIBERO_WRITABLE_ROOT/LIBERO"
export LIBERO_CONFIG_PATH="$LIBERO_WRITABLE_ROOT/config"
export WAN_VAE_PATH="$COSMOS_WRITABLE_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

export COSMOS_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export LIBERO_PYTHON="$LIBERO_WRITABLE_ROOT/envs/libero-eval/bin/python"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

export LD_LIBRARY_PATH=
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MUJOCO_GL=egl
