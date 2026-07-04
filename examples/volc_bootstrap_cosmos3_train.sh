#!/usr/bin/env bash
# Source-safe environment bootstrap for VolcEngine Cosmos3 training jobs.
# This file is intentionally small and non-mutating: it must not install packages,
# edit shell startup files, or assume an interactive shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file is meant to be sourced, not executed." >&2
  exit 2
fi

COSMOS3_FRAMEWORK_HOME="${COSMOS3_FRAMEWORK_HOME:-/mlp_vepfs/share/swy/cosmos3-framework}"
DEXMAL_ROOT="${DEXMAL_ROOT:-/dexmal-datainfra-swy}"
WANDB_TOKEN_PATH="${WANDB_TOKEN_PATH:-$DEXMAL_ROOT/bootstrap/wandb_token}"

export COSMOS3_FRAMEWORK_HOME
export HF_HOME="${HF_HOME:-$COSMOS3_FRAMEWORK_HOME/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$COSMOS3_FRAMEWORK_HOME/venvs/cosmos-framework-cu130-train}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$COSMOS3_FRAMEWORK_HOME/uv/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$COSMOS3_FRAMEWORK_HOME/uv/cache}"
export PATH="$COSMOS3_FRAMEWORK_HOME/uv/bin:${PATH:-}"

if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_TOKEN_PATH" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_TOKEN_PATH")"
  export WANDB_API_KEY
fi

# NGC/PyTorch images can pick up incompatible host CUDA libraries from this.
export LD_LIBRARY_PATH=
