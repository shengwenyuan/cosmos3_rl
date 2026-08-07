#!/usr/bin/env bash
# Local single-node 8xL20X wrapper for full Cosmos3-DROID action-policy SFT.
#
# Goal: reproduce the Cosmos3-Nano-Policy-DROID post-training route as closely
# as the released assets allow on this PAI DSW machine. The wrapper delegates
# actual training to the official paired launcher:
#
#   examples/launch_sft_action_policy_droid.sh
#
# Paper alignment defaults:
# - success split only, matching failure-demonstration removal
# - DROID LeRobot v3.0, 15 FPS, 360x640 source videos
# - policy mode, raw 8D absolute joint-position actions + initial state
# - 32 future actions and auxiliary RGB video frames
# - 480p concat view -> 540x640
# - Cosmos3-Nano DCP resume, Wan2.2 VAE, official DROID recipe hyperparams
# - idle-frame filtering is required by default, but the public
#   Cosmos3-DROID repository does not include the filter JSON. Set
#   FILTER_DICT_PATH once that artifact is available, or explicitly set
#   REQUIRE_FILTER_DICT=0 USE_FILTER_DICT=0 to run the released raw success
#   split instead.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/cosmos-framework}"
FAST_ROOT="${FAST_ROOT:-/mlp_vepfs/share/swy/cosmos3-framework}"
TRAIN_VENV="${TRAIN_VENV:-$FAST_ROOT/venvs/cosmos-framework-cu130-train}"

DATASET_PATH="${DATASET_PATH:-$FAST_ROOT/modelscope/datasets/Cosmos3-DROID/success}"
DROID_ROOT="${DROID_ROOT:-$DATASET_PATH}"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$FAST_ROOT/checkpoints/Cosmos3-Nano-dcp}"
WAN_VAE_PATH="${WAN_VAE_PATH:-$FAST_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
QWEN_TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-$FAST_ROOT/modelscope/hub/Qwen/Qwen3-VL-8B-Instruct}"

JOB_NAME="${JOB_NAME:-droid_full_001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FAST_ROOT/outputs/droid_full_bs32}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$OUTPUT_ROOT}"
LOG_FILENAME="${LOG_FILENAME:-${JOB_NAME}_sft.log}"

TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_droid_repro.toml}"
LAUNCHER="${LAUNCHER:-examples/launch_sft_action_policy_droid.sh}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TOKEN_PATH="${WANDB_TOKEN_PATH:-/dexmal-datainfra-swy/bootstrap/wandb_token}"
ALLOW_WANDB_MISSING="${ALLOW_WANDB_MISSING:-0}"
FIX_WANDB_CORE="${FIX_WANDB_CORE:-1}"

RUN_DRYRUN_FIRST="${RUN_DRYRUN_FIRST:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRYRUN_ONLY="${DRYRUN_ONLY:-0}"
RUN_VIDEO_SAMPLE_CHECK="${RUN_VIDEO_SAMPLE_CHECK:-1}"
RUN_TOKENIZER_CHECK="${RUN_TOKENIZER_CHECK:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-50013}"
DP_SHARD="${DP_SHARD:-8}"
DP_REPLICATE="${DP_REPLICATE:-1}"

# The official recipe uses 128 per rank and was validated on larger multi-node
# systems. The current 8xL20X run used 4 with ~43.9 GiB / 143.8 GiB per GPU, so
# 32 is a practical full-DROID starting point on this node. Reduce to 24/16 on
# OOM; raise only after a smoke run proves memory headroom.
MAX_SAMPLES_PER_BATCH="${MAX_SAMPLES_PER_BATCH:-32}"

# Empty means keep the official TOML values: max_iter=10000, save_iter=1000,
# logging_iter=50, scheduler.cycle_lengths=[10000].
MAX_ITER="${MAX_ITER:-}"
SAVE_ITER="${SAVE_ITER:-}"
LOGGING_ITER="${LOGGING_ITER:-}"

# The paper says community-provided idle-frame filtering was applied. The public
# ModelScope/HF Cosmos3-DROID repository currently exposes only raw success and
# failure LeRobot splits, not the keep-ranges JSON.
USE_FILTER_DICT="${USE_FILTER_DICT:-1}"
REQUIRE_FILTER_DICT="${REQUIRE_FILTER_DICT:-1}"
FILTER_DICT_PATH="${FILTER_DICT_PATH:-}"

EXPECTED_GPU_NAME_SUBSTR="${EXPECTED_GPU_NAME_SUBSTR:-L20X}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-120}"
SKIP_HARDWARE_CHECK="${SKIP_HARDWARE_CHECK:-0}"
EXPECT_FULL_DROID_SUCCESS="${EXPECT_FULL_DROID_SUCCESS:-1}"

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

prepend_ld_library_path() {
  local dir="$1"
  [[ -d "$dir" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$dir:"*) ;;
    *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
}

setup_library_path() {
  local site_packages python_lib_dir torch_lib_dir cu13_lib_dir

  site_packages="$("$PYTHON_BIN" - <<'PY_SITE'
import site
print(site.getsitepackages()[0])
PY_SITE
)"
  python_lib_dir="$("$PYTHON_BIN" - <<'PY_LIB'
from pathlib import Path
import sysconfig
print(Path(sysconfig.get_paths()["stdlib"]).parent)
PY_LIB
)"

  cu13_lib_dir="$site_packages/nvidia/cu13/lib"
  torch_lib_dir="$site_packages/torch/lib"
  prepend_ld_library_path "$cu13_lib_dir"
  prepend_ld_library_path "$python_lib_dir"
  prepend_ld_library_path "$torch_lib_dir"
}

activate_repo_env() {
  if [[ -r "$HOME/.bashrc" ]]; then
    source_env_file "$HOME/.bashrc"
  fi

  require_dir "$REPO_ROOT"
  cd "$REPO_ROOT"
  require_file "$TRAIN_VENV/bin/activate"
  # shellcheck disable=SC1090
  source "$TRAIN_VENV/bin/activate"
  hash -r
  PYTHON_BIN="${PYTHON_BIN:-$TRAIN_VENV/bin/python}"
  require_file "$PYTHON_BIN"
  export PYTHON_BIN
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  setup_library_path
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC PYTORCH_ALLOC_CONF NCCL_DEBUG
}

validate_static_values() {
  local name value
  for name in JOB_NAME NPROC_PER_NODE MASTER_PORT DATASET_PATH DROID_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN_TOKENIZER_PATH; do
    value="${!name-}"
    [[ "$value" != *"<"* && "$value" != *">"* && "$value" != "TODO"* && "$value" != "todo"* ]] \
      || die "$name still looks like a placeholder: $value"
  done

  [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || die "invalid NPROC_PER_NODE=$NPROC_PER_NODE"
  [[ "$DP_SHARD" =~ ^[0-9]+$ ]] || die "invalid DP_SHARD=$DP_SHARD"
  [[ "$DP_REPLICATE" =~ ^[0-9]+$ ]] || die "invalid DP_REPLICATE=$DP_REPLICATE"
  [[ "$MAX_SAMPLES_PER_BATCH" =~ ^[0-9]+$ ]] || die "invalid MAX_SAMPLES_PER_BATCH=$MAX_SAMPLES_PER_BATCH"
  [[ "$USE_FILTER_DICT" =~ ^[01]$ ]] || die "USE_FILTER_DICT must be 0 or 1"
  [[ "$REQUIRE_FILTER_DICT" =~ ^[01]$ ]] || die "REQUIRE_FILTER_DICT must be 0 or 1"
  [[ "$EXPECT_FULL_DROID_SUCCESS" =~ ^[01]$ ]] || die "EXPECT_FULL_DROID_SUCCESS must be 0 or 1"
  [[ -z "$MAX_ITER" || "$MAX_ITER" =~ ^[0-9]+$ ]] || die "invalid MAX_ITER=$MAX_ITER"
  [[ -z "$SAVE_ITER" || "$SAVE_ITER" =~ ^[0-9]+$ ]] || die "invalid SAVE_ITER=$SAVE_ITER"
  [[ -z "$LOGGING_ITER" || "$LOGGING_ITER" =~ ^[0-9]+$ ]] || die "invalid LOGGING_ITER=$LOGGING_ITER"
  (( NPROC_PER_NODE == 8 )) || die "this local L20X wrapper expects NPROC_PER_NODE=8, got $NPROC_PER_NODE"
  (( DP_SHARD * DP_REPLICATE == NPROC_PER_NODE )) \
    || die "DP_SHARD * DP_REPLICATE must equal NPROC_PER_NODE for single-node launch"
  [[ "$DROID_ROOT" == */success ]] || die "paper-aligned launcher expects the success split, got DROID_ROOT=$DROID_ROOT"
}

setup_logging() {
  mkdir -p "$OUTPUT_ROOT/orchestrator_logs"
  ORCH_LOG="$OUTPUT_ROOT/orchestrator_logs/${JOB_NAME}.$(hostname).log"
  export ORCH_LOG
  exec > >(tee -a "$ORCH_LOG") 2>&1
}

resolve_filter_dict_path() {
  if [[ -n "$FILTER_DICT_PATH" ]]; then
    return 0
  fi

  local candidate
  for candidate in \
    "$FAST_ROOT/modelscope/datasets/Cosmos3-DROID/keep_ranges_1_0_1.json" \
    "$FAST_ROOT/modelscope/datasets/Cosmos3-DROID/success/keep_ranges_1_0_1.json" \
    "$FAST_ROOT/modelscope/datasets/Cosmos3-DROID/filter/keep_ranges_1_0_1.json" \
    "$REPO_ROOT/tmps/keep_ranges_1_0_1.json"; do
    if [[ -f "$candidate" ]]; then
      FILTER_DICT_PATH="$candidate"
      return 0
    fi
  done
}

validate_filter_plan() {
  resolve_filter_dict_path

  if [[ "$USE_FILTER_DICT" == "1" ]]; then
    if [[ -z "$FILTER_DICT_PATH" || ! -f "$FILTER_DICT_PATH" ]]; then
      if [[ "$REQUIRE_FILTER_DICT" == "1" ]]; then
        die "paper-aligned DROID run requires idle-frame keep-ranges filter JSON, but FILTER_DICT_PATH is missing. The public Cosmos3-DROID repository does not include it; set FILTER_DICT_PATH once obtained, or run raw success split with REQUIRE_FILTER_DICT=0 USE_FILTER_DICT=0."
      fi
      log "WARNING: USE_FILTER_DICT=1 requested but FILTER_DICT_PATH is missing; disabling filter because REQUIRE_FILTER_DICT=0."
      USE_FILTER_DICT=0
    else
      require_file "$FILTER_DICT_PATH"
    fi
  fi
}

validate_paths() {
  require_dir "$FAST_ROOT"
  require_file "$TOML_FILE"
  require_file "$LAUNCHER"
  require_dir "$TRAIN_VENV"
  require_file "$TRAIN_VENV/bin/python"
  require_cmd ffmpeg

  require_dir "$DROID_ROOT"
  require_file "$DROID_ROOT/meta/info.json"
  require_file "$DROID_ROOT/meta/tasks.parquet"
  require_dir "$DROID_ROOT/data"
  require_dir "$DROID_ROOT/videos"
  require_dir "$DROID_ROOT/videos/observation.image.exterior_image_1_left"
  require_dir "$DROID_ROOT/videos/observation.image.exterior_image_2_left"
  require_dir "$DROID_ROOT/videos/observation.image.wrist_image_left"

  require_dir "$BASE_CHECKPOINT_PATH"
  require_file "$BASE_CHECKPOINT_PATH/checkpoint.json"
  require_file "$BASE_CHECKPOINT_PATH/model/.metadata"
  local shard_count
  shard_count="$(find "$BASE_CHECKPOINT_PATH/model" -maxdepth 1 -name '*.distcp' | wc -l | tr -d ' ')"
  (( shard_count > 0 )) || die "no .distcp shards under $BASE_CHECKPOINT_PATH/model"

  require_file "$WAN_VAE_PATH"
  require_dir "$QWEN_TOKENIZER_PATH"
  require_file "$QWEN_TOKENIZER_PATH/tokenizer_config.json"
  require_file "$QWEN_TOKENIZER_PATH/tokenizer.json"
  require_file "$QWEN_TOKENIZER_PATH/vocab.json"
  require_file "$QWEN_TOKENIZER_PATH/merges.txt"
}

validate_dataset_metadata() {
  EXPECT_FULL_DROID_SUCCESS="$EXPECT_FULL_DROID_SUCCESS" \
  "$PYTHON_BIN" - "$DROID_ROOT/meta/info.json" <<'PY_DATASET'
import json
import os
import sys

info = json.load(open(sys.argv[1]))
features = info.get("features", {})
errors = []
if info.get("codebase_version") != "v3.0":
    errors.append(f"codebase_version={info.get('codebase_version')!r}, expected 'v3.0'")
if int(info.get("fps", -1)) != 15:
    errors.append(f"fps={info.get('fps')!r}, expected 15")
if os.environ.get("EXPECT_FULL_DROID_SUCCESS") == "1":
    expected_counts = {
        "total_episodes": 57639,
        "total_frames": 18691281,
        "total_tasks": 53086,
    }
    for key, expected in expected_counts.items():
        actual = int(info.get(key, -1))
        if actual != expected:
            errors.append(f"{key}={actual}, expected full success split value {expected}")
expected = {
    "observation.image.exterior_image_1_left": [360, 640, 3],
    "observation.image.exterior_image_2_left": [360, 640, 3],
    "observation.image.wrist_image_left": [360, 640, 3],
    "observation.state.joint_positions": [7],
    "observation.state.gripper_position": [1],
    "action.joint_position": [7],
    "action.gripper_position": [1],
}
for key, shape in expected.items():
    actual = features.get(key, {}).get("shape")
    if actual != shape:
        errors.append(f"{key} shape={actual!r}, expected {shape!r}")
if errors:
    print("Cosmos3-DROID metadata validation failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    raise SystemExit(1)
print(
    "Cosmos3-DROID metadata ok:",
    f"episodes={info.get('total_episodes')}",
    f"frames={info.get('total_frames')}",
    f"tasks={info.get('total_tasks')}",
    f"fps={info.get('fps')}",
)
print("DROID action contract: raw 8D [joint7, gripper] + use_state=True -> 33x8 action window")
print("DROID video contract: wrist top row + left/right shoulder bottom row concat_view")
PY_DATASET
}

validate_video_sample() {
  if [[ "$RUN_VIDEO_SAMPLE_CHECK" != "1" ]]; then
    log "Skipping video sample decode because RUN_VIDEO_SAMPLE_CHECK=$RUN_VIDEO_SAMPLE_CHECK."
    return 0
  fi

  "$PYTHON_BIN" - "$DROID_ROOT" "$USE_FILTER_DICT" "${FILTER_DICT_PATH:-}" <<'PY_SAMPLE'
import sys
from cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

root, use_filter, filter_path = sys.argv[1], sys.argv[2] == "1", sys.argv[3] or None
ds = DROIDLeRobotDataset(
    root=root,
    fps=15.0,
    chunk_length=32,
    action_space="joint_pos",
    mode="policy",
    use_state=True,
    viewpoint="concat_view",
    use_image_augmentation=False,
    use_filter_dict=use_filter,
    filter_dict_path=filter_path,
    action_normalization=None,
)
print("DROID sample index ok:", f"len={len(ds)}", f"shuffle_blocks={len(ds.get_shuffle_blocks())}", f"use_filter_dict={use_filter}")
sample = ds[0]
video_shape = tuple(sample["video"].shape)
action_shape = tuple(sample["action"].shape)
print("DROID sample decode ok:", f"video={video_shape}/{sample['video'].dtype}", f"action={action_shape}/{sample['action'].dtype}")
if video_shape != (3, 33, 540, 640):
    raise SystemExit(f"unexpected concat_view video shape: {video_shape}")
if action_shape != (33, 8):
    raise SystemExit(f"unexpected joint_pos action shape: {action_shape}")
PY_SAMPLE
}

validate_tokenizer() {
  if [[ "$RUN_TOKENIZER_CHECK" != "1" ]]; then
    log "Skipping tokenizer check because RUN_TOKENIZER_CHECK=$RUN_TOKENIZER_CHECK."
    return 0
  fi

  "$PYTHON_BIN" - "$QWEN_TOKENIZER_PATH" <<'PY_TOKENIZER'
import sys
from cosmos_framework.configs.base.defaults.vlm import create_qwen2_tokenizer_with_download

processor = create_qwen2_tokenizer_with_download(sys.argv[1], "hf")
print("Qwen tokenizer ok:", type(processor).__name__, sys.argv[1])
PY_TOKENIZER
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
import sys
import torch

expected_name = os.environ.get("EXPECTED_GPU_NAME_SUBSTR", "")
min_mem_gb = float(os.environ.get("MIN_GPU_MEMORY_GB", "0"))
print("python", sys.executable)
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

reject_filter_conflicts() {
  local overrides="$1"
  if [[ "$overrides" == *"use_filter_dict"* || "$overrides" == *"filter_dict_path"* ]]; then
    die "sample-filter override detected in EXTRA_TAIL_OVERRIDES; use USE_FILTER_DICT/FILTER_DICT_PATH instead"
  fi
}

build_tail_overrides() {
  reject_filter_conflicts "$USER_EXTRA_TAIL_OVERRIDES"

  TAIL_OVERRIDES=(
    "job.name=$JOB_NAME"
    "job.wandb_mode=$WANDB_MODE"
    "model.config.vlm_config.tokenizer.pretrained_model_name=$QWEN_TOKENIZER_PATH"
    "dataloader_train.max_samples_per_batch=$MAX_SAMPLES_PER_BATCH"
    "model.config.parallelism.data_parallel_shard_degree=$DP_SHARD"
    "model.config.parallelism.data_parallel_replicate_degree=$DP_REPLICATE"
  )

  if [[ "$USE_FILTER_DICT" == "1" ]]; then
    TAIL_OVERRIDES+=(
      "dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=True"
      "dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path=$FILTER_DICT_PATH"
    )
  else
    TAIL_OVERRIDES+=("dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=False")
  fi

  if [[ -n "$MAX_ITER" ]]; then
    TAIL_OVERRIDES+=("trainer.max_iter=$MAX_ITER" "scheduler.cycle_lengths=[$MAX_ITER]")
  fi
  if [[ -n "$SAVE_ITER" ]]; then
    TAIL_OVERRIDES+=("checkpoint.save_iter=$SAVE_ITER")
  fi
  if [[ -n "$LOGGING_ITER" ]]; then
    TAIL_OVERRIDES+=("trainer.logging_iter=$LOGGING_ITER")
  fi

  if [[ -n "$USER_EXTRA_TAIL_OVERRIDES" ]]; then
    EXTRA_TAIL_OVERRIDES="${TAIL_OVERRIDES[*]} ${USER_EXTRA_TAIL_OVERRIDES}"
  else
    EXTRA_TAIL_OVERRIDES="${TAIL_OVERRIDES[*]}"
  fi
  export EXTRA_TAIL_OVERRIDES
}

log_effective_plan() {
  log "Effective local L20X full Cosmos3-DROID launch plan:"
  log "  JOB_NAME=$JOB_NAME"
  log "  TRAIN_VENV=$TRAIN_VENV"
  log "  PYTHON_BIN=$PYTHON_BIN"
  log "  NPROC_PER_NODE=$NPROC_PER_NODE MASTER_PORT=$MASTER_PORT"
  log "  DP_SHARD=$DP_SHARD DP_REPLICATE=$DP_REPLICATE"
  log "  MAX_SAMPLES_PER_BATCH=$MAX_SAMPLES_PER_BATCH"
  log "  MAX_ITER=${MAX_ITER:-TOML default 10000}"
  log "  SAVE_ITER=${SAVE_ITER:-TOML default 1000}"
  log "  LOGGING_ITER=${LOGGING_ITER:-TOML default 50}"
  log "  DATASET_PATH=$DATASET_PATH"
  log "  DROID_ROOT=$DROID_ROOT"
  log "  USE_FILTER_DICT=$USE_FILTER_DICT FILTER_DICT_PATH=${FILTER_DICT_PATH:-unset}"
  log "  BASE_CHECKPOINT_PATH=$BASE_CHECKPOINT_PATH"
  log "  WAN_VAE_PATH=$WAN_VAE_PATH"
  log "  QWEN_TOKENIZER_PATH=$QWEN_TOKENIZER_PATH"
  log "  OUTPUT_ROOT=$OUTPUT_ROOT"
  log "  EXTRA_TAIL_OVERRIDES=$EXTRA_TAIL_OVERRIDES"
}

run_dryrun() {
  if [[ "$RUN_DRYRUN_FIRST" != "1" ]]; then
    log "Skipping direct config dryrun because RUN_DRYRUN_FIRST=$RUN_DRYRUN_FIRST."
    return 0
  fi
  log "Running direct config dryrun before torchrun."
  IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT/dryrun_${JOB_NAME}" \
  DATASET_PATH="$DATASET_PATH" \
  DROID_ROOT="$DROID_ROOT" \
  BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  "$PYTHON_BIN" -m cosmos_framework.scripts.train \
    --dryrun \
    --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}"
}

launch_training() {
  export DATASET_PATH DROID_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH QWEN_TOKENIZER_PATH
  export OUTPUT_ROOT IMAGINAIRE_OUTPUT_ROOT LOG_FILENAME
  export NPROC_PER_NODE MASTER_PORT PYTHON_BIN
  log "Launching full Cosmos3-DROID action-policy SFT through the official paired launcher."
  bash "$LAUNCHER"
}

main() {
  activate_repo_env
  validate_static_values
  setup_logging

  log "Starting local L20X full Cosmos3-DROID action-policy training wrapper."
  validate_filter_plan
  validate_paths
  validate_dataset_metadata
  validate_video_sample
  validate_tokenizer
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
