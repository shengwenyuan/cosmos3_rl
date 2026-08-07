#!/usr/bin/env bash
# Minimal Alibaba PAI-DLC multi-node smoke test.
#
# No dataset, checkpoint, W&B, or Cosmos training config is required. Each DLC
# worker runs this same script. It resolves worker topology, checks that the
# requested 8 nodes * 8 GPUs layout is visible, then launches a tiny NCCL
# all-reduce through torchrun.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_NAME="${JOB_NAME:-paidlc_topology_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/paidlc_topology_smoke}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${PAI_HPS_NPROC_PER_NODE:-8}}"
NNODES="${NNODES:-${PAI_HPS_NNODES:-}}"
NODE_RANK="${NODE_RANK:-${PAI_HPS_NODE_RANK:-}}"
MASTER_ADDR="${MASTER_ADDR:-${PAI_HPS_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-${PAI_HPS_MASTER_PORT:-50012}}"

EXPECTED_NNODES="${EXPECTED_NNODES:-8}"
EXPECTED_NPROC_PER_NODE="${EXPECTED_NPROC_PER_NODE:-8}"
STRICT_EXPECTED_TOPOLOGY="${STRICT_EXPECTED_TOPOLOGY:-1}"
TOPOLOGY_ONLY="${TOPOLOGY_ONLY:-0}"

NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
SMOKE_TIMEOUT_SEC="${SMOKE_TIMEOUT_SEC:-600}"

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
  log "Smoke log: ${SMOKE_LOG:-unset}"
  exit "$exit_code"
}
trap on_error ERR

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

activate_python() {
  if [[ -r "$HOME/.bashrc" ]]; then
    source_env_file "$HOME/.bashrc"
  fi

  cd "$REPO_ROOT"
  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    die "python is not available"
  fi

  export PYTHON_BIN
  export LD_LIBRARY_PATH=
  export NCCL_DEBUG TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC SMOKE_TIMEOUT_SEC
}

resolve_topology() {
  [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || die "invalid NPROC_PER_NODE=$NPROC_PER_NODE"
  [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "invalid MASTER_PORT=$MASTER_PORT"

  if [[ -z "$NNODES" ]]; then
    if NNODES="$(first_set PAI_WORKER_NUM PAI_WORKER_COUNT MLP_WORKER_NUM VC_WORKER_NUM)"; then
      :
    elif [[ -n "${WORLD_SIZE:-}" && "$WORLD_SIZE" =~ ^[0-9]+$ && "$WORLD_SIZE" -eq "$EXPECTED_NNODES" ]]; then
      # Some PyTorchJob launchers expose WORLD_SIZE as the number of node
      # replicas before torchrun starts local GPU processes.
      NNODES="$WORLD_SIZE"
    elif [[ -n "${WORLD_SIZE:-}" && "$WORLD_SIZE" =~ ^[0-9]+$ && "$WORLD_SIZE" -ge "$NPROC_PER_NODE" && $((WORLD_SIZE % NPROC_PER_NODE)) -eq 0 ]]; then
      NNODES=$((WORLD_SIZE / NPROC_PER_NODE))
    else
      NNODES=1
    fi
  fi

  if [[ -z "$NODE_RANK" ]]; then
    if NODE_RANK="$(first_set PAI_WORKER_INDEX PAI_WORKER_RANK PAI_CURRENT_TASK_INDEX PAI_TASK_INDEX PAI_ROLE_INDEX MLP_ROLE_INDEX MLP_TASK_INDEX VC_TASK_INDEX)"; then
      :
    elif [[ -n "${RANK:-}" && "$RANK" =~ ^[0-9]+$ && -n "${WORLD_SIZE:-}" && "$WORLD_SIZE" =~ ^[0-9]+$ && "$WORLD_SIZE" -eq "$EXPECTED_NNODES" ]]; then
      # In that same launcher mode, RANK is the node replica rank.
      NODE_RANK="$RANK"
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
  if [[ "$STRICT_EXPECTED_TOPOLOGY" == "1" ]]; then
    (( NNODES == EXPECTED_NNODES )) || die "expected NNODES=$EXPECTED_NNODES, got $NNODES"
    (( NPROC_PER_NODE == EXPECTED_NPROC_PER_NODE )) || die "expected NPROC_PER_NODE=$EXPECTED_NPROC_PER_NODE, got $NPROC_PER_NODE"
  fi

  export NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT WORLD_SIZE
}

setup_logging() {
  mkdir -p "$OUTPUT_ROOT"
  SMOKE_LOG="$OUTPUT_ROOT/${JOB_NAME}.node${NODE_RANK}.$(hostname).log"
  export SMOKE_LOG
  exec > >(tee -a "$SMOKE_LOG") 2>&1
}

print_plan() {
  log "PAI-DLC topology smoke:"
  log "  JOB_NAME=$JOB_NAME"
  log "  REPO_ROOT=$REPO_ROOT"
  log "  PYTHON_BIN=$PYTHON_BIN"
  log "  NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE WORLD_SIZE=$WORLD_SIZE"
  log "  MASTER_ADDR=${MASTER_ADDR:-<unset>} MASTER_PORT=$MASTER_PORT"
  log "  OUTPUT_ROOT=$OUTPUT_ROOT"
  log "  TOPOLOGY_ONLY=$TOPOLOGY_ONLY"
  log "  NCCL_DEBUG=$NCCL_DEBUG"
}

write_smoke_program() {
  SMOKE_PY="$(mktemp /tmp/paidlc_smoke.XXXXXX.py)"
  export SMOKE_PY
  cat > "$SMOKE_PY" <<'PY_SMOKE'
import datetime
import os
import socket
import sys

import torch
import torch.distributed as dist


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ["WORLD_SIZE"])
timeout = int(os.environ.get("SMOKE_TIMEOUT_SEC", "600"))
host = socket.gethostname()

if not torch.cuda.is_available():
    fail("CUDA is not available")

device_count = torch.cuda.device_count()
if local_rank >= device_count:
    fail(f"LOCAL_RANK={local_rank} but cuda_device_count={device_count}")

torch.cuda.set_device(local_rank)
device_name = torch.cuda.get_device_name(local_rank)
print(
    f"rank={rank}/{world_size} local_rank={local_rank} host={host} "
    f"cuda_count={device_count} device={device_name}",
    flush=True,
)

dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(seconds=timeout),
)

x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

expected = world_size * (world_size + 1) / 2.0
actual = float(x.item())
if abs(actual - expected) > 0.5:
    fail(f"all_reduce mismatch: actual={actual} expected={expected}")

payload = f"rank={rank} host={host} local_rank={local_rank} device={device_name}"
gathered = [None for _ in range(world_size)] if rank == 0 else None
dist.gather_object(payload, gathered, dst=0)
dist.barrier()

if rank == 0:
    print(f"PAI-DLC smoke all_reduce OK: world_size={world_size} sum={actual:.1f}", flush=True)
    for item in gathered:
        print(f"  {item}", flush=True)

dist.destroy_process_group()
PY_SMOKE
}

run_smoke() {
  write_smoke_program
  trap 'rm -f "${SMOKE_PY:-}"' EXIT

  local torchrun_args=(
    --nproc_per_node="$NPROC_PER_NODE"
    --nnodes="$NNODES"
    --node_rank="$NODE_RANK"
    --master_port="$MASTER_PORT"
  )
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    torchrun_args+=(--master_addr="$MASTER_ADDR")
  fi

  log "Launching torchrun smoke."
  "$PYTHON_BIN" -m torch.distributed.run "${torchrun_args[@]}" "$SMOKE_PY"
}

main() {
  activate_python
  resolve_topology
  setup_logging
  print_plan

  if [[ "$TOPOLOGY_ONLY" == "1" ]]; then
    log "Topology-only check passed."
    return 0
  fi

  run_smoke
  log "PAI-DLC topology smoke passed."
}

main "$@"
