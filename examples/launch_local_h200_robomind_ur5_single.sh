#!/usr/bin/env bash
# Local single-node 8xH200 wrapper for RoboMIND1-UR single-arm joint-pos SFT.
#
# This is the standalone-machine analogue of the cloud-facing H20 wrappers:
# it validates local paths / hardware, prepares run overrides, optionally runs
# a direct config dryrun, then delegates to the canonical paired launcher:
#
#   examples/launch_sft_action_policy_robomind_ur5_single.sh
#
# It intentionally does not source or configure any Volcengine / MLP service.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/cosmos-framework}"
FAST_ROOT="${FAST_ROOT:-/mlp_vepfs/share/swy/cosmos3-framework}"

DATASET_PATH="${DATASET_PATH:-$FAST_ROOT/lerobot/RoboMIND1-ur5}"
UR5_SINGLE_ROOT="${UR5_SINGLE_ROOT:-$DATASET_PATH}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$FAST_ROOT/checkpoints/Cosmos3-Nano-dcp}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$FAST_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_NAME="${JOB_NAME:-action_policy_robomind_ur5_single_h200_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FAST_ROOT/outputs/robomind_ur5_single_h200}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
LOG_FILENAME="${LOG_FILENAME:-${JOB_NAME}_sft.log}"

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml}"
LAUNCHER="${LAUNCHER:-examples/launch_sft_action_policy_robomind_ur5_single.sh}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TOKEN_PATH="${WANDB_TOKEN_PATH:-/dexmal-datainfra-swy/bootstrap/wandb_token}"
ALLOW_WANDB_MISSING="${ALLOW_WANDB_MISSING:-0}"
FIX_WANDB_CORE="${FIX_WANDB_CORE:-1}"

RUN_DRYRUN_FIRST="${RUN_DRYRUN_FIRST:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRYRUN_ONLY="${DRYRUN_ONLY:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-50012}"
DP_SHARD="${DP_SHARD:-8}"
DP_REPLICATE="${DP_REPLICATE:-1}"

MAX_ITER="${MAX_ITER:-10000}"
SAVE_ITER="${SAVE_ITER:-1000}"
LOGGING_ITER="${LOGGING_ITER:-50}"
# H200 should normally have enough memory for the recipe's per-rank batch.
# If OOM, lower this first, e.g. MAX_SAMPLES_PER_BATCH=64 or 32.
MAX_SAMPLES_PER_BATCH="${MAX_SAMPLES_PER_BATCH:-128}"
# Full recipe LR 2e-4 targets global batch 8192; 8xH200 with batch 128 is 1024.
OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"
SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH:-$MAX_ITER}"

EXPECTED_GPU_NAME_SUBSTR="${EXPECTED_GPU_NAME_SUBSTR:-H200}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-120}"
SKIP_HARDWARE_CHECK="${SKIP_HARDWARE_CHECK:-0}"

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

USER_EXTRA_TAIL_OVERRIDES="${EXTRA_TAIL_OVERRIDES:-}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

on_error() {
  local exit_code=$?
  log "FAILED with exit code ${exit_code} at line ${BASH_LINENO[0]:-unknown}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
  log "Orchestration log: ${ORCH_LOG:-unset}"
  log "Training log, if launched: ${OUTPUT_ROOT}/logs/${LOG_FILENAME}"
  exit "$exit_code"
}
trap on_error ERR

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

source_env_file() {
  local env_file="$1"
  local had_nounset=0

  case "$-" in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac

  # shellcheck disable=SC1090
  source "$env_file"

  if (( had_nounset )); then
    set -u
  fi
}

activate_repo_env() {
  if [[ -r "$HOME/.bashrc" ]]; then
    source_env_file "$HOME/.bashrc"
  fi

  require_dir "$REPO_ROOT"
  cd "$REPO_ROOT"
  require_file ".venv/bin/activate"
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  hash -r
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  export PYTHON_BIN
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH=
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC PYTORCH_ALLOC_CONF NCCL_DEBUG
}

validate_static_values() {
  local name value
  for name in JOB_NAME NPROC_PER_NODE MASTER_PORT DATASET_PATH UR5_SINGLE_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH; do
    value="${!name-}"
    [[ "$value" != *"<"* && "$value" != *">"* && "$value" != "TODO"* && "$value" != "todo"* ]] \
      || die "$name still looks like a placeholder: $value"
  done

  [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || die "invalid NPROC_PER_NODE=$NPROC_PER_NODE"
  [[ "$DP_SHARD" =~ ^[0-9]+$ ]] || die "invalid DP_SHARD=$DP_SHARD"
  [[ "$DP_REPLICATE" =~ ^[0-9]+$ ]] || die "invalid DP_REPLICATE=$DP_REPLICATE"
  [[ "$MAX_SAMPLES_PER_BATCH" =~ ^[0-9]+$ ]] || die "invalid MAX_SAMPLES_PER_BATCH=$MAX_SAMPLES_PER_BATCH"
  (( NPROC_PER_NODE == 8 )) || die "this local H200 wrapper expects NPROC_PER_NODE=8, got $NPROC_PER_NODE"
  (( DP_SHARD * DP_REPLICATE == NPROC_PER_NODE )) \
    || die "DP_SHARD * DP_REPLICATE must equal NPROC_PER_NODE for single-node launch"
}

setup_logging() {
  mkdir -p "$OUTPUT_ROOT/orchestrator_logs"
  ORCH_LOG="$OUTPUT_ROOT/orchestrator_logs/${JOB_NAME}.$(hostname).log"
  export ORCH_LOG
  exec > >(tee -a "$ORCH_LOG") 2>&1
}

validate_paths() {
  require_dir "$FAST_ROOT"
  require_file "$TOML_FILE"
  require_file "$LAUNCHER"

  require_dir "$UR5_SINGLE_ROOT"
  require_file "$UR5_SINGLE_ROOT/meta/info.json"
  require_dir "$UR5_SINGLE_ROOT/data"
  require_dir "$UR5_SINGLE_ROOT/videos"

  require_dir "$BASE_CHECKPOINT_PATH"
  require_file "$BASE_CHECKPOINT_PATH/checkpoint.json"
  require_file "$BASE_CHECKPOINT_PATH/model/.metadata"
  local shard_count
  shard_count="$(find "$BASE_CHECKPOINT_PATH/model" -maxdepth 1 -name '*.distcp' | wc -l | tr -d ' ')"
  (( shard_count > 0 )) || die "no .distcp shards under $BASE_CHECKPOINT_PATH/model"

  require_file "$WAN_VAE_PATH"
}

validate_dataset_metadata() {
  "$PYTHON_BIN" - "$UR5_SINGLE_ROOT/meta/info.json" <<'PY_DATASET'
import json
import sys

info = json.load(open(sys.argv[1]))
features = info.get("features", {})
errors = []
if info.get("codebase_version") != "v3.0":
    errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v3.0'")
if int(info.get("fps", -1)) != 15:
    errors.append(f"fps={info.get('fps')!r}, expected 15")
expected = {
    "action.arm_left_joint": [6],
    "action.gripper_left": [1],
    "observation.state.arm_left_joint": [6],
    "observation.state.gripper_left": [1],
}
for key, shape in expected.items():
    actual = features.get(key, {}).get("shape")
    if actual != shape:
        errors.append(f"{key} shape={actual!r}, expected {shape!r}")
if "observation.images.camera_top" not in features:
    errors.append("missing camera feature observation.images.camera_top")
if errors:
    print("RoboMIND1-UR metadata validation failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(1)
print(
    "RoboMIND1-UR metadata ok:",
    f"episodes={info.get('total_episodes')}",
    f"frames={info.get('total_frames')}",
    f"fps={info.get('fps')}",
)
print("UR5 action contract: raw 7D [joint6, gripper] + use_state=True")
print("UR5 canvas contract: camera_top as first real view; missing views zero-padded")
PY_DATASET
}

validate_hardware() {
  if [[ "$SKIP_HARDWARE_CHECK" == "1" ]]; then
    log "Skipping hardware check because SKIP_HARDWARE_CHECK=1."
    return 0
  fi

  require_cmd nvidia-smi
  nvidia-smi
  EXPECTED_GPU_NAME_SUBSTR="$EXPECTED_GPU_NAME_SUBSTR" \
  MIN_GPU_MEMORY_GB="$MIN_GPU_MEMORY_GB" \
  "$PYTHON_BIN" - <<'PY_HARDWARE'
import os
import torch

expected_name = os.environ.get("EXPECTED_GPU_NAME_SUBSTR", "")
min_mem_gb = float(os.environ.get("MIN_GPU_MEMORY_GB", "0"))
print("python", os.sys.executable)
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected exactly 8 CUDA devices, got {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f"gpu[{i}] {props.name} mem_gb={mem_gb:.1f}")
    if expected_name and expected_name not in props.name:
        raise SystemExit(f"gpu[{i}] name {props.name!r} does not contain {expected_name!r}")
    if mem_gb < min_mem_gb:
        raise SystemExit(f"gpu[{i}] memory {mem_gb:.1f}GB < required {min_mem_gb:.1f}GB")
PY_HARDWARE
}

validate_wandb() {
  if [[ "$WANDB_MODE" == "disabled" ]]; then
    log "W&B disabled by WANDB_MODE=disabled."
    return 0
  fi

  if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_TOKEN_PATH" ]]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_TOKEN_PATH")"
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    [[ "$ALLOW_WANDB_MISSING" == "1" ]] || die "WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is missing"
    log "WARNING: WANDB_API_KEY missing; continuing because ALLOW_WANDB_MISSING=1."
  fi

  local wandb_core
  wandb_core="$("$PYTHON_BIN" - <<'PY_WANDB'
from pathlib import Path
try:
    import wandb
    print(Path(wandb.__file__).resolve().parent / "bin" / "wandb-core")
except Exception:
    print("")
PY_WANDB
)"
  if [[ -n "$wandb_core" && -f "$wandb_core" && ! -x "$wandb_core" ]]; then
    if [[ "$FIX_WANDB_CORE" == "1" ]]; then
      chmod u+x "$wandb_core" || die "failed to chmod +x $wandb_core"
      log "Fixed W&B core executable bit: $wandb_core"
    else
      die "W&B core is not executable: $wandb_core"
    fi
  fi
}

write_job_metadata() {
  local meta_dir="$OUTPUT_ROOT/job_meta/$JOB_NAME"
  mkdir -p "$meta_dir"
  env | sort > "$meta_dir/env.$(hostname).txt"
  git status --short > "$meta_dir/git_status.$(hostname).txt" || true
  git rev-parse HEAD > "$meta_dir/git_head.$(hostname).txt" || true
  nvidia-smi -q > "$meta_dir/nvidia_smi.$(hostname).txt" || true
}

build_tail_overrides() {
  TAIL_OVERRIDES=(
    "job.name=$JOB_NAME"
    "job.wandb_mode=$WANDB_MODE"
    "trainer.max_iter=$MAX_ITER"
    "trainer.logging_iter=$LOGGING_ITER"
    "checkpoint.save_iter=$SAVE_ITER"
    "dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH"
    "optimizer.lr=$OPTIMIZER_LR"
    "scheduler.cycle_lengths=[$SCHEDULER_CYCLE_LENGTH]"
    "model.config.parallelism.data_parallel_shard_degree=$DP_SHARD"
    "model.config.parallelism.data_parallel_replicate_degree=$DP_REPLICATE"
    "trainer.profiling.target_ranks=[0]"
  )

  if [[ -n "$USER_EXTRA_TAIL_OVERRIDES" ]]; then
    EXTRA_TAIL_OVERRIDES="${TAIL_OVERRIDES[*]} ${USER_EXTRA_TAIL_OVERRIDES}"
  else
    EXTRA_TAIL_OVERRIDES="${TAIL_OVERRIDES[*]}"
  fi
  export EXTRA_TAIL_OVERRIDES
}

log_effective_plan() {
  log "Effective local H200 launch plan:"
  log "  JOB_NAME=$JOB_NAME"
  log "  NPROC_PER_NODE=$NPROC_PER_NODE MASTER_PORT=$MASTER_PORT"
  log "  DP_SHARD=$DP_SHARD DP_REPLICATE=$DP_REPLICATE"
  log "  MAX_SAMPLES_PER_BATCH=$MAX_SAMPLES_PER_BATCH OPTIMIZER_LR=$OPTIMIZER_LR"
  log "  DATASET_PATH=$DATASET_PATH"
  log "  UR5_SINGLE_ROOT=$UR5_SINGLE_ROOT"
  log "  BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
  log "  WAN_VAE_PATH=$WAN_VAE_PATH"
  log "  OUTPUT_ROOT=$OUTPUT_ROOT"
  log "  ACTION_CONTRACT=raw 7D [joint6,gripper] + use_state=True; gripper polarity intentionally unchanged"
  log "  EXTRA_TAIL_OVERRIDES=$EXTRA_TAIL_OVERRIDES"
}

run_dryrun() {
  if [[ "$RUN_DRYRUN_FIRST" != "1" ]]; then
    log "Skipping direct config dryrun because RUN_DRYRUN_FIRST=$RUN_DRYRUN_FIRST."
    return 0
  fi
  log "Running direct config dryrun before torchrun."
  IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT/dryrun_${JOB_NAME}" \
  UR5_SINGLE_ROOT="$UR5_SINGLE_ROOT" \
  BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  "$PYTHON_BIN" -m cosmos_framework.scripts.train \
    --dryrun \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}" trainer.max_iter=1
}

launch_training() {
  export DATASET_PATH UR5_SINGLE_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH
  export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT LOG_FILENAME
  export NPROC_PER_NODE MASTER_PORT
  log "Launching RoboMIND1-UR single-arm joint-pos SFT."
  bash "$LAUNCHER"
}

main() {
  activate_repo_env
  validate_static_values
  setup_logging

  log "Starting local H200 RoboMIND1-UR single-arm joint-pos training wrapper."
  validate_paths
  validate_dataset_metadata
  validate_hardware
  validate_wandb
  write_job_metadata
  build_tail_overrides
  log_effective_plan

  if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    log "Preflight checks passed; exiting before dryrun/training because PREFLIGHT_ONLY=1."
    return 0
  fi

  run_dryrun
  if [[ "$DRYRUN_ONLY" == "1" ]]; then
    log "Dryrun passed; exiting before torchrun because DRYRUN_ONLY=1."
    return 0
  fi

  launch_training

  local run_dir="$OUTPUT_ROOT/cosmos3_action/action_sft/$JOB_NAME"
  log "Training command exited successfully."
  log "Run dir: $run_dir"
  if [[ -f "$run_dir/checkpoints/latest_checkpoint.txt" ]]; then
    log "Latest checkpoint: $(cat "$run_dir/checkpoints/latest_checkpoint.txt")"
  fi
}

main "$@"
