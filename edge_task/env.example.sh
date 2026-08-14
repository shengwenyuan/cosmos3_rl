#!/usr/bin/env bash
# Copy into the task platform's private environment configuration.
# Never add HF_TOKEN or other secrets to this file.

export COSMOS_REPO_ROOT=/root/workspace/cosmos-framework
export COSMOS_WRITABLE_ROOT=/data/cosmos-edge-libero

export UV_CACHE_DIR="$COSMOS_WRITABLE_ROOT/cache/uv"
export HF_HOME="$COSMOS_WRITABLE_ROOT/cache/huggingface"
export IMAGINAIRE_OUTPUT_ROOT="$COSMOS_WRITABLE_ROOT/runs"

export EDGE_BASE_PATH="$COSMOS_WRITABLE_ROOT/models/Cosmos3-Edge"
export EDGE_DCP_PATH="$COSMOS_WRITABLE_ROOT/checkpoints/Cosmos3-Edge-DCP"
export LIBERO_ROOT="$COSMOS_WRITABLE_ROOT/datasets/LIBERO_LeRobot_v3"
export WAN_VAE_PATH=/mnt/bos/1011/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

export COSMOS_PYTHON=/opt/cosmos-venv/bin/python
export LIBERO_PYTHON=/opt/libero-venv/bin/python

export LD_LIBRARY_PATH=
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export MUJOCO_GL=egl
