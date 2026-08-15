#!/usr/bin/env bash
# Read-only Cosmos3 Edge-LIBERO audit. It never downloads assets or writes to /mnt.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${COSMOS_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
FAILURES=0
WARNINGS=0

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        pass "command $1 -> $(command -v "$1")"
    else
        fail "command $1 is missing"
    fi
}

printf 'Cosmos3 Edge Policy LIBERO environment audit\n'
printf 'repo=%s\n' "$REPO_ROOT"

for command_name in uv ffmpeg git-lfs; do
    check_command "$command_name"
done

if [[ -f "$REPO_ROOT/.python-version" ]]; then
    pass ".python-version=$(tr -d '[:space:]' < "$REPO_ROOT/.python-version")"
else
    fail "$REPO_ROOT/.python-version is missing"
fi

if [[ -n "${COSMOS_PYTHON:-}" && -x "$COSMOS_PYTHON" ]]; then
    MAIN_PYTHON="$COSMOS_PYTHON"
elif [[ -x /opt/cosmos-venv/bin/python ]]; then
    MAIN_PYTHON=/opt/cosmos-venv/bin/python
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    MAIN_PYTHON="$REPO_ROOT/.venv/bin/python"
else
    MAIN_PYTHON=""
    fail "Cosmos training Python is missing"
fi

if [[ -n "$MAIN_PYTHON" ]]; then
    "$MAIN_PYTHON" --version || fail "Cosmos Python cannot start"
fi

for variable_name in UV_CACHE_DIR HF_HOME IMAGINAIRE_OUTPUT_ROOT; do
    variable_value="${!variable_name:-}"
    if [[ -z "$variable_value" ]]; then
        fail "$variable_name is unset"
    elif [[ -d "$variable_value" && -w "$variable_value" ]]; then
        pass "$variable_name is writable: $variable_value"
    else
        fail "$variable_name is not an existing writable directory: $variable_value"
    fi
done

if [[ "${HF_ENDPOINT:-}" == "https://huggingface.co" ]]; then
    pass "HF_ENDPOINT is pinned to the user-approved official Hugging Face endpoint"
else
    fail "HF_ENDPOINT must be exactly https://huggingface.co"
fi

if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -d "$UV_PROJECT_ENVIRONMENT" ]]; then
    pass "persistent uv project environment: $UV_PROJECT_ENVIRONMENT"
else
    fail "UV_PROJECT_ENVIRONMENT is unset or missing"
fi

if [[ "${UV_LINK_MODE:-}" == "hardlink" ]]; then
    pass "UV_LINK_MODE=hardlink for same-CFS cache and venv"
else
    warn "UV_LINK_MODE is not hardlink; CFS installation may be very slow"
fi

VAE_PATH="${WAN_VAE_PATH:-/mnt/cfs/data/swy/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
if [[ -r "$VAE_PATH" ]]; then
    VAE_SIZE="$(stat -c '%s' "$VAE_PATH" 2>/dev/null || true)"
    if [[ "$VAE_SIZE" == "2818839170" ]]; then
        pass "Wan2.2 VAE is readable and size matches: $VAE_PATH"
    else
        warn "Wan2.2 VAE is readable but size is unexpected: ${VAE_SIZE:-unknown}"
    fi
else
    fail "Wan2.2 VAE is not readable: $VAE_PATH"
fi

STATS_PATH="$REPO_ROOT/cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
if [[ -r "$STATS_PATH" ]]; then
    pass "LIBERO rot6d normalizer is readable"
else
    fail "LIBERO rot6d normalizer is missing: $STATS_PATH"
fi

for recipe_path in \
    "$REPO_ROOT/cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_all_edge.py" \
    "$REPO_ROOT/examples/toml/sft_config/action_policy_libero_all_edge_8gpu.toml" \
    "$REPO_ROOT/examples/toml/sft_config/action_policy_libero_all_edge_4gpu.toml" \
    "$REPO_ROOT/examples/launch_sft_action_policy_libero_all_edge_8gpu.sh" \
    "$REPO_ROOT/examples/launch_sft_action_policy_libero_all_edge_4gpu.sh"; do
    if [[ -r "$recipe_path" ]]; then
        pass "Edge-LIBERO recipe component: ${recipe_path#"$REPO_ROOT/"}"
    else
        fail "Edge-LIBERO recipe component is missing: $recipe_path"
    fi
done

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L | wc -l)"
    pass "nvidia-smi sees $GPU_COUNT GPU(s)"
else
    fail "nvidia-smi cannot see a GPU"
fi

if [[ -n "$MAIN_PYTHON" ]]; then
    if "$MAIN_PYTHON" - <<'PY'
from multiprocessing import shared_memory
x = shared_memory.SharedMemory(create=True, size=64)
x.close()
x.unlink()
PY
    then
        pass "POSIX shared memory create/unlink"
    else
        fail "POSIX shared memory is not writable"
    fi

    if "$MAIN_PYTHON" - <<'PY'
import importlib.util
required = [
    "torch", "torchvision", "transformers", "diffusers", "av", "pyarrow",
    "lerobot", "cosmos_framework", "omegaconf", "tyro",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing imports: " + ", ".join(missing))
PY
    then
        pass "Cosmos training modules are discoverable"
    else
        fail "Cosmos training modules are incomplete"
    fi

    if "$MAIN_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert torch.cuda.device_count() > 0, "no CUDA devices"
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
assert (x @ x).isfinite().all()
print(f"torch={torch.__version__} cuda={torch.version.cuda} devices={torch.cuda.device_count()}")
PY
    then
        pass "PyTorch CUDA BF16 smoke"
    else
        fail "PyTorch CUDA BF16 smoke failed"
    fi
fi

BASE_PATH="${EDGE_BASE_PATH:-}"
if [[ -z "$BASE_PATH" ]]; then
    fail "EDGE_BASE_PATH is unset"
elif [[ -f "$BASE_PATH/config.json" && -f "$BASE_PATH/model.safetensors.index.json" ]]; then
    pass "Edge base checkpoint structure: $BASE_PATH"
else
    fail "Edge base checkpoint is incomplete: $BASE_PATH"
fi

DCP_PATH="${EDGE_DCP_PATH:-}"
if [[ -z "$DCP_PATH" ]]; then
    fail "EDGE_DCP_PATH is unset"
elif [[ -f "$DCP_PATH/model/.metadata" && -f "$DCP_PATH/model/config.json" ]]; then
    pass "Edge DCP structure: $DCP_PATH"
else
    fail "Edge DCP is incomplete: $DCP_PATH"
fi

DATA_ROOT="${LIBERO_ROOT:-/mnt/cfs/data/swy/libero/LIBERO_LeRobot_v3}"
for suite_name in libero_10 libero_goal libero_object libero_spatial; do
    suite_path="$DATA_ROOT/$suite_name"
    if [[ -f "$suite_path/meta/info.json" && -d "$suite_path/data" && -d "$suite_path/videos" ]]; then
        pass "LIBERO suite structure: $suite_name"
    else
        fail "LIBERO suite is incomplete: $suite_path"
    fi
done

if [[ -n "${LIBERO_PYTHON:-}" && -x "$LIBERO_PYTHON" ]]; then
    SIM_PYTHON="$LIBERO_PYTHON"
elif [[ -x /opt/libero-venv/bin/python ]]; then
    SIM_PYTHON=/opt/libero-venv/bin/python
else
    SIM_PYTHON=""
    fail "LIBERO simulator Python is missing"
fi

if [[ -n "$SIM_PYTHON" ]]; then
    if MUJOCO_GL="${MUJOCO_GL:-egl}" "$SIM_PYTHON" - <<'PY'
import libero
import mujoco
import robosuite
print("libero=ok", "mujoco=" + mujoco.__version__, "robosuite=" + robosuite.__version__)
PY
    then
        pass "LIBERO simulator imports"
    else
        fail "LIBERO simulator imports failed"
    fi
fi

printf 'SUMMARY failures=%d warnings=%d\n' "$FAILURES" "$WARNINGS"
if (( FAILURES > 0 )); then
    exit 1
fi
