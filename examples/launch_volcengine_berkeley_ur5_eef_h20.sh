#!/usr/bin/env bash
# Volcengine / multi-H20 orchestration wrapper for Berkeley AUTOLab UR5 EEF SFT.
#
# This is the outer, cloud-job-facing entry point. It validates the mounted
# datasets/checkpoints, sources the dev-image environment, checks hardware and
# W&B, computes distributed topology overrides, then delegates to the canonical
# paired launcher:
#
#   examples/launch_sft_action_policy_berkeley_ur5_eef.sh
#
# Cost guard: Berkeley EEF frame canonicalization has not been FK-validated yet.
# Set ALLOW_UNVERIFIED_EEF=1 only when intentionally running with the current
# provisional [dx,dy,dz,droll,dpitch,dyaw,gripper] mapping.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/cosmos-framework}"
FAST_ROOT="${FAST_ROOT:-/mlp_vepfs/share/swy/cosmos3-framework}"
SLOW_ROOT="${SLOW_ROOT:-/dexmal-datainfra-swy}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-$SLOW_ROOT/bootstrap}"
VOLC_BOOTSTRAP_FILE="${VOLC_BOOTSTRAP_FILE:-$REPO_ROOT/examples/volc_bootstrap_cosmos3_train.sh}"

DATASET_PATH="${DATASET_PATH:-$FAST_ROOT/lerobot/berkeley_autolab_ur5}"
BERKELEY_UR5_ROOT="${BERKELEY_UR5_ROOT:-$DATASET_PATH}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$FAST_ROOT/checkpoints/Cosmos3-Nano-dcp}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$FAST_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_NAME="${JOB_NAME:-action_policy_berkeley_ur5_eef_h20_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FAST_ROOT/outputs/berkeley_ur5_eef_h20}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
LOG_FILENAME="${LOG_FILENAME:-${JOB_NAME}_sft.log}"

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_berkeley_ur5_eef_repro.toml}"
WANDB_MODE="${WANDB_MODE:-online}"
FIX_WANDB_CORE="${FIX_WANDB_CORE:-1}"
ALLOW_WANDB_MISSING="${ALLOW_WANDB_MISSING:-0}"
ALLOW_UNVERIFIED_EEF="${ALLOW_UNVERIFIED_EEF:-0}"
RUN_DRYRUN_FIRST="${RUN_DRYRUN_FIRST:-1}"

DP_SHARD="${DP_SHARD:-8}"
RECOMMENDED_WORLD_SIZE="${RECOMMENDED_WORLD_SIZE:-64}"
MIN_WORLD_SIZE="${MIN_WORLD_SIZE:-8}"
STRICT_RECOMMENDED_WORLD_SIZE="${STRICT_RECOMMENDED_WORLD_SIZE:-0}"

MAX_ITER="${MAX_ITER:-3000}"
SAVE_ITER="${SAVE_ITER:-500}"
LOGGING_ITER="${LOGGING_ITER:-50}"
MAX_SAMPLES_PER_BATCH="${MAX_SAMPLES_PER_BATCH:-128}"
MASTER_PORT="${MASTER_PORT:-50012}"
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

source_env() {
  [[ -r "$VOLC_BOOTSTRAP_FILE" ]] || die "VOLC_BOOTSTRAP_FILE is not readable: $VOLC_BOOTSTRAP_FILE"
  source_env_file "$VOLC_BOOTSTRAP_FILE"

  # dev-machine-bootstrap.sh is not source-safe; it installs packages and edits
  # ~/.bashrc. Source only explicitly provided env files after the stable
  # VolcEngine bootstrap, so cluster/job-specific overrides win.
  if [[ -n "${COSMOS3_EXTRA_ENV_FILE:-}" ]]; then
    [[ -r "$COSMOS3_EXTRA_ENV_FILE" ]] || die "COSMOS3_EXTRA_ENV_FILE is not readable: $COSMOS3_EXTRA_ENV_FILE"
    source_env_file "$COSMOS3_EXTRA_ENV_FILE"
  elif [[ -r "$BOOTSTRAP_DIR/cosmos3-train-env.sh" ]]; then
    source_env_file "$BOOTSTRAP_DIR/cosmos3-train-env.sh"
  fi

  export LD_LIBRARY_PATH=
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC PYTORCH_ALLOC_CONF NCCL_DEBUG
}

activate_repo_env() {
  require_dir "$REPO_ROOT"
  cd "$REPO_ROOT"
  require_file ".venv/bin/activate"
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
}

detect_nproc_per_node() {
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    printf '%s\n' "$NPROC_PER_NODE"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
    return
  fi
  python - <<'PY_DETECT_GPU'
import torch
print(torch.cuda.device_count())
PY_DETECT_GPU
}

resolve_topology() {
  NPROC_PER_NODE="$(detect_nproc_per_node)"
  [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || die "invalid NPROC_PER_NODE=$NPROC_PER_NODE"
  (( NPROC_PER_NODE > 0 )) || die "NPROC_PER_NODE must be > 0"

  NNODES="${NNODES:-${MLP_WORKER_NUM:-${VC_WORKER_NUM:-1}}}"
  NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-${MLP_TASK_INDEX:-${VC_TASK_INDEX:-0}}}}"
  [[ "$NNODES" =~ ^[0-9]+$ ]] || die "invalid NNODES=$NNODES"
  [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "invalid NODE_RANK=$NODE_RANK"
  (( NNODES > 0 )) || die "NNODES must be > 0"
  (( NODE_RANK < NNODES )) || die "NODE_RANK=$NODE_RANK must be < NNODES=$NNODES"

  if [[ -z "${MASTER_ADDR:-}" && "$NNODES" -gt 1 ]]; then
    if [[ -n "${MLP_WORKER_0_HOST:-}" ]]; then
      MASTER_ADDR="$MLP_WORKER_0_HOST"
    elif [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
      MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
    else
      die "MASTER_ADDR is required for NNODES=$NNODES; export MASTER_ADDR to rank-0 host/IP"
    fi
  fi

  WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
  (( WORLD_SIZE >= MIN_WORLD_SIZE )) || die "WORLD_SIZE=$WORLD_SIZE is below MIN_WORLD_SIZE=$MIN_WORLD_SIZE"
  if (( WORLD_SIZE < RECOMMENDED_WORLD_SIZE )); then
    log "WARNING: WORLD_SIZE=$WORLD_SIZE is below recommended $RECOMMENDED_WORLD_SIZE for the current full-batch recipe."
    [[ "$STRICT_RECOMMENDED_WORLD_SIZE" != "1" ]] || die "STRICT_RECOMMENDED_WORLD_SIZE=1 rejects WORLD_SIZE=$WORLD_SIZE"
  fi

  (( DP_SHARD > 0 )) || die "DP_SHARD must be > 0"
  (( WORLD_SIZE % DP_SHARD == 0 )) || die "WORLD_SIZE=$WORLD_SIZE must be divisible by DP_SHARD=$DP_SHARD"
  DP_REPLICATE="${DP_REPLICATE:-$((WORLD_SIZE / DP_SHARD))}"
  (( DP_SHARD * DP_REPLICATE == WORLD_SIZE )) || die "DP_SHARD * DP_REPLICATE must equal WORLD_SIZE"

  export NPROC_PER_NODE NNODES NODE_RANK MASTER_PORT WORLD_SIZE DP_REPLICATE
  [[ -n "${MASTER_ADDR:-}" ]] && export MASTER_ADDR
}

setup_logging() {
  mkdir -p "$OUTPUT_ROOT/orchestrator_logs"
  ORCH_LOG="$OUTPUT_ROOT/orchestrator_logs/${JOB_NAME}.node${NODE_RANK:-0}.$(hostname).log"
  export ORCH_LOG
  exec > >(tee -a "$ORCH_LOG") 2>&1
}

validate_paths() {
  require_dir "$FAST_ROOT"
  require_dir "$SLOW_ROOT"
  require_dir "$BOOTSTRAP_DIR"

  require_dir "$BERKELEY_UR5_ROOT"
  require_file "$BERKELEY_UR5_ROOT/meta/info.json"
  require_dir "$BERKELEY_UR5_ROOT/data"
  require_dir "$BERKELEY_UR5_ROOT/videos"

  require_dir "$BASE_CHECKPOINT_PATH"
  require_file "$BASE_CHECKPOINT_PATH/checkpoint.json"
  require_file "$BASE_CHECKPOINT_PATH/model/.metadata"
  local shard_count
  shard_count="$(find "$BASE_CHECKPOINT_PATH/model" -maxdepth 1 -name '*.distcp' | wc -l | tr -d ' ')"
  (( shard_count > 0 )) || die "no .distcp shards under $BASE_CHECKPOINT_PATH/model"

  require_file "$WAN_VAE_PATH"
  require_file "$TOML_FILE"
  require_file "examples/launch_sft_action_policy_berkeley_ur5_eef.sh"
}

validate_dataset_metadata() {
  python - "$BERKELEY_UR5_ROOT/meta/info.json" <<'PY_DATASET'
import json
import sys

info = json.load(open(sys.argv[1]))
features = info.get("features", {})
errors = []
if info.get("codebase_version") != "v3.0":
    errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v3.0'")
if int(info.get("fps", -1)) != 5:
    errors.append(f"fps={info.get('fps')!r}, expected 5")
if features.get("action", {}).get("shape") != [7]:
    errors.append(f"action shape={features.get('action', {}).get('shape')!r}, expected [7]")
for key in ("observation.images.image", "observation.images.hand_image"):
    if key not in features:
        errors.append(f"missing camera feature {key}")
if errors:
    print("Berkeley metadata validation failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(1)
print(
    "Berkeley metadata ok:",
    f"episodes={info.get('total_episodes')}",
    f"frames={info.get('total_frames')}",
    f"fps={info.get('fps')}",
)
PY_DATASET
}

validate_hardware() {
  require_cmd nvidia-smi
  require_cmd python
  nvidia-smi
  python - <<'PY_HARDWARE'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"gpu[{i}] {props.name} mem_gb={props.total_memory / 1024**3:.1f}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY_HARDWARE
  local local_gpu_count
  local_gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
  (( local_gpu_count >= NPROC_PER_NODE )) || die "local GPU count $local_gpu_count < NPROC_PER_NODE=$NPROC_PER_NODE"
}

validate_wandb() {
  if [[ "$WANDB_MODE" == "disabled" ]]; then
    log "W&B disabled by WANDB_MODE=disabled."
    return
  fi

  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if [[ -r "${WANDB_TOKEN_PATH:-$BOOTSTRAP_DIR/wandb_token}" ]]; then
      export WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_TOKEN_PATH:-$BOOTSTRAP_DIR/wandb_token}")"
    fi
  fi

  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    [[ "$ALLOW_WANDB_MISSING" == "1" ]] || die "WANDB_MODE=$WANDB_MODE but WANDB_API_KEY is missing"
    log "WARNING: WANDB_API_KEY missing; continuing because ALLOW_WANDB_MISSING=1"
  fi

  local wandb_core
  wandb_core="$(python - <<'PY_WANDB'
from pathlib import Path
try:
    import wandb
    p = Path(wandb.__file__).resolve().parent / "bin" / "wandb-core"
    print(p)
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
  env | sort > "$meta_dir/env.node${NODE_RANK}.$(hostname).txt"
  git status --short > "$meta_dir/git_status.node${NODE_RANK}.$(hostname).txt" || true
  nvidia-smi -q > "$meta_dir/nvidia_smi.node${NODE_RANK}.$(hostname).txt" || true
}

build_tail_overrides() {
  TAIL_OVERRIDES=(
    "job.name=$JOB_NAME"
    "job.wandb_mode=$WANDB_MODE"
    "trainer.max_iter=$MAX_ITER"
    "trainer.logging_iter=$LOGGING_ITER"
    "checkpoint.save_iter=$SAVE_ITER"
    "dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH"
    "model.config.parallelism.data_parallel_shard_degree=$DP_SHARD"
    "model.config.parallelism.data_parallel_replicate_degree=$DP_REPLICATE"
    "trainer.profiling.target_ranks=[0]"
  )

  if [[ -n "${OPTIMIZER_LR:-}" ]]; then
    TAIL_OVERRIDES+=("optimizer.lr=$OPTIMIZER_LR")
  fi
  if [[ -n "${SCHEDULER_CYCLE_LENGTH:-}" ]]; then
    TAIL_OVERRIDES+=("scheduler.cycle_lengths=[$SCHEDULER_CYCLE_LENGTH]")
  fi
  if [[ -n "$USER_EXTRA_TAIL_OVERRIDES" ]]; then
    # The downstream launcher consumes EXTRA_TAIL_OVERRIDES with shell word
    # splitting, so keep user-supplied overrides space-separated as before.
    EXTRA_TAIL_OVERRIDES="${TAIL_OVERRIDES[*]} ${USER_EXTRA_TAIL_OVERRIDES}"
  else
    EXTRA_TAIL_OVERRIDES="${TAIL_OVERRIDES[*]}"
  fi
  export EXTRA_TAIL_OVERRIDES
}

run_dryrun() {
  [[ "$RUN_DRYRUN_FIRST" == "1" ]] || return
  log "Running direct config dryrun before torchrun."
  IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT/dryrun_${JOB_NAME}" \
  BERKELEY_UR5_ROOT="$BERKELEY_UR5_ROOT" \
  BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  python -m cosmos_framework.scripts.train \
    --dryrun \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}" trainer.max_iter=1
}

launch_training() {
  export DATASET_PATH BERKELEY_UR5_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH
  export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT LOG_FILENAME
  log "Launching Berkeley UR5 EEF SFT."
  log "JOB_NAME=$JOB_NAME"
  log "WORLD_SIZE=$WORLD_SIZE NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE"
  log "DP_SHARD=$DP_SHARD DP_REPLICATE=$DP_REPLICATE"
  log "DATASET_PATH=$DATASET_PATH"
  log "BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
  log "OUTPUT_ROOT=$OUTPUT_ROOT"
  log "EXTRA_TAIL_OVERRIDES=$EXTRA_TAIL_OVERRIDES"
  bash examples/launch_sft_action_policy_berkeley_ur5_eef.sh
}

main() {
  source_env
  activate_repo_env
  resolve_topology
  setup_logging

  log "Starting Volcengine Berkeley UR5 EEF orchestration."
  if [[ "$ALLOW_UNVERIFIED_EEF" != "1" ]]; then
    die "Berkeley EEF frame is still provisional; set ALLOW_UNVERIFIED_EEF=1 to acknowledge and launch."
  fi

  validate_paths
  validate_dataset_metadata
  validate_hardware
  validate_wandb
  write_job_metadata
  build_tail_overrides
  run_dryrun
  launch_training

  local run_dir="$OUTPUT_ROOT/cosmos3_action/action_sft/$JOB_NAME"
  log "Training command exited successfully."
  log "Run dir: $run_dir"
  if [[ -f "$run_dir/checkpoints/latest_checkpoint.txt" ]]; then
    log "Latest checkpoint: $(cat "$run_dir/checkpoints/latest_checkpoint.txt")"
  fi
}

main "$@"
