#!/usr/bin/env bash
# Shared PAI DSW adapter for a canonical UR5 action-policy launcher.

set -euo pipefail

if [[ -r "$HOME/.bashrc" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$HOME/.bashrc"
    set -u
fi

REPO_ROOT="${REPO_ROOT:-/root/code/cosmos-framework}"
FAST_ROOT="${FAST_ROOT:-/mlp_vepfs/share/swy/cosmos3-framework}"
TRAIN_VENV="${TRAIN_VENV:-$FAST_ROOT/venvs/cosmos-framework-cu130-train}"
PYTHON_BIN="${PYTHON_BIN:-$TRAIN_VENV/bin/python}"

: "${DATASET_PATH:?DATASET_PATH must be set by the recipe wrapper}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$FAST_ROOT/checkpoints/Cosmos3-Nano-dcp}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$FAST_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
QWEN_TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-$FAST_ROOT/modelscope/hub/Qwen/Qwen3-VL-8B-Instruct}"

: "${JOB_NAME:?JOB_NAME must be set by the recipe wrapper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FAST_ROOT/outputs/$JOB_NAME}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
ADOPT_LEGACY_ACTION_POLICY_MANIFEST="${ADOPT_LEGACY_ACTION_POLICY_MANIFEST:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-50012}"
: "${LAUNCHER:?LAUNCHER must be set by the recipe wrapper}"

[[ -d "$REPO_ROOT" ]] || { echo "ERROR: missing repo: $REPO_ROOT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$QWEN_TOKENIZER_PATH" ]] || { echo "ERROR: missing tokenizer: $QWEN_TOKENIZER_PATH" >&2; exit 1; }
cd "$REPO_ROOT"

SITE_PACKAGES="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
PYTHON_LIB_DIR="$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.base_prefix) / "lib")')"
export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$SITE_PACKAGES/nvidia/cu13/lib:$SITE_PACKAGES/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action/action_sft/$JOB_NAME"
if [[ -e "$RUN_DIR" && "$ALLOW_RESUME" != "1" ]]; then
    echo "ERROR: run already exists; choose JOB_NAME/OUTPUT_ROOT or set ALLOW_RESUME=1: $RUN_DIR" >&2
    exit 1
fi

EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:+$EXTRA_TAIL_OVERRIDES }job.name=$JOB_NAME model.config.vlm_config.tokenizer.pretrained_model_name=$QWEN_TOKENIZER_PATH"
export PYTHON_BIN PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_PATH BASE_CHECKPOINT_PATH WAN_VAE_PATH
export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT NPROC_PER_NODE MASTER_PORT EXTRA_TAIL_OVERRIDES
export ADOPT_LEGACY_ACTION_POLICY_MANIFEST

echo ">>> Launching $JOB_NAME from $REPO_ROOT"
exec bash "$LAUNCHER"
