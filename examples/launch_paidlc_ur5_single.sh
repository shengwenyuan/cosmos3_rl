#!/usr/bin/env bash
# Alibaba PAI-DLC multi-node wrapper for RoboMIND1-UR single-arm joint-pos SFT.
#
# This is the cloud-job-facing entry point for 8 nodes * 8 L20X GPUs. It
# validates mounted paths and local hardware, resolves the PAI-DLC worker
# topology into torchrun's NNODES/NODE_RANK/MASTER_ADDR contract, prepares SFT
# tail overrides, then delegates to:
#
#   examples/launch_sft_action_policy_robomind_ur5_single.sh
#
# PAI reference:
#   https://help.aliyun.com/zh/pai/developer-reference/submit-a-training-job

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/cosmos-framework}"
FAST_ROOT="${FAST_ROOT:-${COSMOS3_FRAMEWORK_HOME:-/mlp_vepfs/share/swy/cosmos3-framework}}"
TRAIN_VENV="${TRAIN_VENV:-$FAST_ROOT/venvs/cosmos-framework-cu130-train}"

DATASET_PATH="${DATASET_PATH:-${PAI_INPUT_TRAIN:-$FAST_ROOT/lerobot/robomind1-ur5-joint}}"
UR5_SINGLE_ROOT="${UR5_SINGLE_ROOT:-$DATASET_PATH}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$FAST_ROOT/checkpoints/Cosmos3-Nano-dcp}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$FAST_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
QWEN_TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-$FAST_ROOT/modelscope/hub/Qwen/Qwen3-VL-8B-Instruct}"

PAI_JOB_TOKEN="${PAI_JOB_ID:-${PAI_JOB_NAME:-${PAI_TRIAL_ID:-}}}"
RUN_STAMP="${RUN_STAMP:-${PAI_JOB_TOKEN:-$(date -u +%Y%m%dT%H%M%SZ)}}"
USER_SET_JOB_NAME=0
if [[ -n "${JOB_NAME:-}" ]]; then
  USER_SET_JOB_NAME=1
fi
JOB_NAME="${JOB_NAME:-action_policy_robomind_ur5_single_paidlc_l20x8n_bs32_${RUN_STAMP}}"
if [[ "${USE_PAI_OUTPUT_ROOT:-0}" == "1" && -n "${PAI_OUTPUT_CHECKPOINTS:-}" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-$PAI_OUTPUT_CHECKPOINTS/cosmos3_outputs}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-$FAST_ROOT/outputs/robomind_ur5_single_paidlc_l20x_8n_bs32}"
fi
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
LOG_FILENAME="${LOG_FILENAME:-}"

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml}"
LAUNCHER="${LAUNCHER:-examples/launch_sft_action_policy_robomind_ur5_single.sh}"

WANDB_MODE="${WANDB_MODE:-${PAI_HPS_WANDB_MODE:-online}}"
WANDB_TOKEN_PATH="${WANDB_TOKEN_PATH:-/dexmal-datainfra-swy/bootstrap/wandb_token}"
ALLOW_WANDB_MISSING="${ALLOW_WANDB_MISSING:-0}"
FIX_WANDB_CORE="${FIX_WANDB_CORE:-1}"

RUN_DRYRUN_FIRST="${RUN_DRYRUN_FIRST:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRYRUN_ONLY="${DRYRUN_ONLY:-0}"
TOPOLOGY_CHECK_ONLY="${TOPOLOGY_CHECK_ONLY:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${PAI_HPS_NPROC_PER_NODE:-8}}"
NNODES="${NNODES:-${PAI_HPS_NNODES:-}}"
NODE_RANK="${NODE_RANK:-${PAI_HPS_NODE_RANK:-}}"
MASTER_ADDR="${MASTER_ADDR:-${PAI_HPS_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-${PAI_HPS_MASTER_PORT:-50012}}"

DP_SHARD="${DP_SHARD:-${PAI_HPS_DP_SHARD:-8}}"
DP_REPLICATE="${DP_REPLICATE:-${PAI_HPS_DP_REPLICATE:-}}"
RECOMMENDED_WORLD_SIZE="${RECOMMENDED_WORLD_SIZE:-64}"
MIN_WORLD_SIZE="${MIN_WORLD_SIZE:-8}"
STRICT_RECOMMENDED_WORLD_SIZE="${STRICT_RECOMMENDED_WORLD_SIZE:-0}"

MAX_ITER="${MAX_ITER:-${PAI_HPS_MAX_ITER:-4000}}"
SAVE_ITER="${SAVE_ITER:-${PAI_HPS_SAVE_ITER:-500}}"
LOGGING_ITER="${LOGGING_ITER:-${PAI_HPS_LOGGING_ITER:-50}}"
MAX_SAMPLES_PER_BATCH="${MAX_SAMPLES_PER_BATCH:-${PAI_HPS_MAX_SAMPLES_PER_BATCH:-32}}"
OPTIMIZER_LR="${OPTIMIZER_LR:-${PAI_HPS_OPTIMIZER_LR:-1.0e-4}}"
SCHEDULER_CYCLE_LENGTH="${SCHEDULER_CYCLE_LENGTH:-${PAI_HPS_SCHEDULER_CYCLE_LENGTH:-$MAX_ITER}}"

EXPECTED_GPU_NAME_SUBSTR="${EXPECTED_GPU_NAME_SUBSTR:-${PAI_HPS_EXPECTED_GPU_NAME_SUBSTR:-L20}}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-${PAI_HPS_MIN_GPU_MEMORY_GB:-120}}"
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
  log "Training log, if launched: ${OUTPUT_ROOT}/logs/${LOG_FILENAME:-unset}"
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

first_set() {
  local name
  for name in "$@"; do
    if [[ -n "${!name:-}" ]]; then
      printf '%s\n' "${!name}"
      return 0
    fi
  done
  return 1
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
  if [[ -r "$HOME/.bashrc" ]]; then
    source_env_file "$HOME/.bashrc"
  fi
}

activate_repo_env() {
  require_dir "$REPO_ROOT"
  cd "$REPO_ROOT"
  require_file "$TRAIN_VENV/bin/activate"
  # shellcheck disable=SC1091
  source "$TRAIN_VENV/bin/activate"
  hash -r
  PYTHON_BIN="${PYTHON_BIN:-$TRAIN_VENV/bin/python}"
  export PYTHON_BIN
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

  local site_packages python_lib_dir
  site_packages="$("$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])')"
  python_lib_dir="$("$PYTHON_BIN" -c 'from pathlib import Path; import sysconfig; print(Path(sysconfig.get_paths()["stdlib"]).parent)')"
  export LD_LIBRARY_PATH="$site_packages/torch/lib:$python_lib_dir:$site_packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC PYTORCH_ALLOC_CONF NCCL_DEBUG
}

resolve_topology() {
  [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || die "invalid NPROC_PER_NODE=$NPROC_PER_NODE"
  [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "invalid MASTER_PORT=$MASTER_PORT"
  (( NPROC_PER_NODE > 0 )) || die "NPROC_PER_NODE must be > 0"

  if [[ -z "$NNODES" ]]; then
    if NNODES="$(first_set PAI_WORKER_NUM PAI_WORKER_COUNT MLP_WORKER_NUM VC_WORKER_NUM)"; then
      :
    elif [[ -n "${WORLD_SIZE:-}" && "$WORLD_SIZE" =~ ^[0-9]+$ && "$WORLD_SIZE" -ge "$NPROC_PER_NODE" && $((WORLD_SIZE % NPROC_PER_NODE)) -eq 0 ]]; then
      NNODES=$((WORLD_SIZE / NPROC_PER_NODE))
    else
      NNODES=1
    fi
  fi

  if [[ -z "$NODE_RANK" ]]; then
    if NODE_RANK="$(first_set PAI_WORKER_INDEX PAI_WORKER_RANK PAI_CURRENT_TASK_INDEX PAI_TASK_INDEX PAI_ROLE_INDEX MLP_ROLE_INDEX MLP_TASK_INDEX VC_TASK_INDEX)"; then
      :
    elif [[ -n "${RANK:-}" && "$RANK" =~ ^[0-9]+$ ]]; then
      NODE_RANK=$((RANK / NPROC_PER_NODE))
    else
      NODE_RANK=0
    fi
  fi

  [[ "$NNODES" =~ ^[0-9]+$ ]] || die "invalid NNODES=$NNODES"
  [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || die "invalid NODE_RANK=$NODE_RANK"
  (( NNODES > 0 )) || die "NNODES must be > 0"
  (( NODE_RANK < NNODES )) || die "NODE_RANK=$NODE_RANK must be < NNODES=$NNODES"

  if [[ -z "$MASTER_ADDR" && "$NNODES" -gt 1 ]]; then
    if MASTER_ADDR="$(first_set PAI_MASTER_ADDR PAI_WORKER_0_HOST PAI_WORKER0_HOST MLP_WORKER_0_HOST)"; then
      :
    elif [[ -n "${PAI_WORKER_HOSTS:-}" ]]; then
      MASTER_ADDR="${PAI_WORKER_HOSTS%%,*}"
    elif [[ -n "${VC_WORKER_HOSTS:-}" ]]; then
      MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
    else
      die "MASTER_ADDR is required for NNODES=$NNODES; export MASTER_ADDR to the rank-0 worker host/IP"
    fi
  fi

  WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
  (( WORLD_SIZE >= MIN_WORLD_SIZE )) || die "WORLD_SIZE=$WORLD_SIZE is below MIN_WORLD_SIZE=$MIN_WORLD_SIZE"
  if (( WORLD_SIZE < RECOMMENDED_WORLD_SIZE )); then
    log "WARNING: WORLD_SIZE=$WORLD_SIZE is below recommended $RECOMMENDED_WORLD_SIZE for this recipe."
    [[ "$STRICT_RECOMMENDED_WORLD_SIZE" != "1" ]] || die "STRICT_RECOMMENDED_WORLD_SIZE=1 rejects WORLD_SIZE=$WORLD_SIZE"
  fi

  [[ "$DP_SHARD" =~ ^[0-9]+$ ]] || die "invalid DP_SHARD=$DP_SHARD"
  (( DP_SHARD > 0 )) || die "DP_SHARD must be > 0"
  (( WORLD_SIZE % DP_SHARD == 0 )) || die "WORLD_SIZE=$WORLD_SIZE must be divisible by DP_SHARD=$DP_SHARD"
  DP_REPLICATE="${DP_REPLICATE:-$((WORLD_SIZE / DP_SHARD))}"
  [[ "$DP_REPLICATE" =~ ^[0-9]+$ ]] || die "invalid DP_REPLICATE=$DP_REPLICATE"
  (( DP_SHARD * DP_REPLICATE == WORLD_SIZE )) || die "DP_SHARD * DP_REPLICATE must equal WORLD_SIZE"

  if [[ "$MAX_SAMPLES_PER_BATCH" == "auto" ]]; then
    MAX_SAMPLES_PER_BATCH=32
  fi
  [[ "$MAX_SAMPLES_PER_BATCH" =~ ^[0-9]+$ ]] || die "invalid MAX_SAMPLES_PER_BATCH=$MAX_SAMPLES_PER_BATCH"

  if [[ -z "$LOG_FILENAME" ]]; then
    LOG_FILENAME="${JOB_NAME}.node${NODE_RANK}.sft.log"
  fi

  if [[ "$NNODES" -gt 1 && -z "$PAI_JOB_TOKEN" && "$USER_SET_JOB_NAME" != "1" ]]; then
    log "WARNING: no PAI job token detected. Make sure JOB_NAME is identical on every worker."
  fi

  export NPROC_PER_NODE NNODES NODE_RANK MASTER_PORT WORLD_SIZE DP_REPLICATE MAX_SAMPLES_PER_BATCH LOG_FILENAME
  if [[ -n "$MASTER_ADDR" ]]; then
    export MASTER_ADDR
  fi
}

validate_static_values() {
  local name value
  for name in JOB_NAME NPROC_PER_NODE NNODES NODE_RANK MASTER_PORT TRAIN_VENV DATASET_PATH UR5_SINGLE_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN_TOKENIZER_PATH OUTPUT_ROOT; do
    value="${!name-}"
    [[ "$value" != *"<"* && "$value" != *">"* && "$value" != "TODO"* && "$value" != "todo"* ]] \
      || die "$name still looks like a placeholder: $value"
  done
}

setup_logging() {
  mkdir -p "$OUTPUT_ROOT/orchestrator_logs"
  ORCH_LOG="$OUTPUT_ROOT/orchestrator_logs/${JOB_NAME}.node${NODE_RANK}.$(hostname).log"
  export ORCH_LOG
  exec > >(tee -a "$ORCH_LOG") 2>&1
}

log_platform_env_summary() {
  local key
  log "Selected PAI-DLC/platform environment:"
  for key in \
    PAI_WORKING_DIR PAI_CONFIG_DIR PAI_INPUT_TRAIN PAI_OUTPUT_MODEL PAI_OUTPUT_CHECKPOINTS \
    PAI_JOB_ID PAI_JOB_NAME PAI_TRIAL_ID PAI_WORKER_NUM PAI_WORKER_COUNT PAI_WORKER_INDEX \
    PAI_WORKER_RANK PAI_CURRENT_TASK_INDEX PAI_TASK_INDEX PAI_ROLE_INDEX PAI_MASTER_ADDR \
    PAI_WORKER_0_HOST PAI_WORKER_HOSTS PAI_USER_ARGS \
    NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT WORLD_SIZE; do
    if [[ -n "${!key:-}" ]]; then
      log "  $key=${!key}"
    fi
  done
}

validate_paths() {
  require_dir "$FAST_ROOT"
  require_file "$TOML_FILE"
  require_file "$LAUNCHER"
  require_dir "$TRAIN_VENV"
  require_file "$TRAIN_VENV/bin/python"
  require_cmd ffmpeg

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
  require_dir "$QWEN_TOKENIZER_PATH"
  require_file "$QWEN_TOKENIZER_PATH/vocab.json"
  require_file "$QWEN_TOKENIZER_PATH/merges.txt"
  require_file "$QWEN_TOKENIZER_PATH/tokenizer_config.json"
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
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  "$PYTHON_BIN" - <<'PY_HARDWARE'
import os
import torch

expected_name = os.environ.get("EXPECTED_GPU_NAME_SUBSTR", "")
min_mem_gb = float(os.environ.get("MIN_GPU_MEMORY_GB", "0"))
expected_count = int(os.environ.get("NPROC_PER_NODE", "8"))
print("python", os.sys.executable)
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if torch.cuda.device_count() != expected_count:
    raise SystemExit(f"expected exactly {expected_count} CUDA devices, got {torch.cuda.device_count()}")
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

validate_video_backend() {
  "$PYTHON_BIN" - <<'PY_VIDEO'
import torchcodec
from torchcodec.decoders import VideoDecoder

print("torchcodec", torchcodec.__version__)
print("video_decoder", VideoDecoder.__name__)
PY_VIDEO
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
  env | sort > "$meta_dir/env.node${NODE_RANK}.$(hostname).txt"
  git status --short > "$meta_dir/git_status.node${NODE_RANK}.$(hostname).txt" || true
  git rev-parse HEAD > "$meta_dir/git_head.node${NODE_RANK}.$(hostname).txt" || true
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
    "optimizer.lr=$OPTIMIZER_LR"
    "scheduler.cycle_lengths=[$SCHEDULER_CYCLE_LENGTH]"
    "model.config.vlm_config.tokenizer.pretrained_model_name=$QWEN_TOKENIZER_PATH"
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
  log "Effective PAI-DLC launch plan:"
  log "  JOB_NAME=$JOB_NAME"
  log "  TRAIN_VENV=$TRAIN_VENV"
  log "  PYTHON_BIN=$PYTHON_BIN"
  log "  WORLD_SIZE=$WORLD_SIZE NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=${MASTER_ADDR:-<unset>} MASTER_PORT=$MASTER_PORT"
  log "  DP_SHARD=$DP_SHARD DP_REPLICATE=$DP_REPLICATE"
  log "  MAX_ITER=$MAX_ITER SAVE_ITER=$SAVE_ITER MAX_SAMPLES_PER_BATCH=$MAX_SAMPLES_PER_BATCH OPTIMIZER_LR=$OPTIMIZER_LR"
  log "  DATASET_PATH=$DATASET_PATH"
  log "  UR5_SINGLE_ROOT=$UR5_SINGLE_ROOT"
  log "  BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
  log "  WAN_VAE_PATH=$WAN_VAE_PATH"
  log "  QWEN_TOKENIZER_PATH=$QWEN_TOKENIZER_PATH"
  log "  OUTPUT_ROOT=$OUTPUT_ROOT"
  log "  LOG_FILENAME=$LOG_FILENAME"
  log "  ACTION_CONTRACT=raw 7D [joint6,gripper] + use_state=True; gripper polarity intentionally unchanged"
  log "  EXTRA_TAIL_OVERRIDES=$EXTRA_TAIL_OVERRIDES"
}

run_dryrun() {
  if [[ "$RUN_DRYRUN_FIRST" != "1" ]]; then
    log "Skipping direct config dryrun because RUN_DRYRUN_FIRST=$RUN_DRYRUN_FIRST."
    return 0
  fi
  log "Running direct config dryrun before torchrun."
  IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT/dryrun_${JOB_NAME}/node${NODE_RANK}" \
  UR5_SINGLE_ROOT="$UR5_SINGLE_ROOT" \
  BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  "$PYTHON_BIN" -m cosmos_framework.scripts.train \
    --dryrun \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}" trainer.max_iter=1
}

launch_training() {
  export DATASET_PATH UR5_SINGLE_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN_TOKENIZER_PATH
  export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT LOG_FILENAME
  export NPROC_PER_NODE NNODES NODE_RANK MASTER_PORT
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    export MASTER_ADDR
  fi
  log "Launching RoboMIND1-UR single-arm joint-pos SFT via PAI-DLC."
  bash "$LAUNCHER"
}

main() {
  source_env
  activate_repo_env
  resolve_topology
  validate_static_values

  if [[ "$TOPOLOGY_CHECK_ONLY" == "1" ]]; then
    log_platform_env_summary
    log "Topology check passed."
    return 0
  fi

  setup_logging
  log "Starting PAI-DLC RoboMIND1-UR single-arm joint-pos training wrapper."
  log_platform_env_summary
  validate_paths
  validate_dataset_metadata
  validate_hardware
  validate_video_backend
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
}

main "$@"
