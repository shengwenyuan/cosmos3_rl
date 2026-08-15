#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=edge_task/env.example.sh
source "$SCRIPT_DIR/env.example.sh"

has_edge_snapshot() {
    local root=$1
    [[ -f "$root/config.json" ]] || return 1
    [[ -f "$root/model.safetensors" || -f "$root/model.safetensors.index.json" ]] && return 0
    [[ -n "$(find "$root" -type f -name '*.safetensors' -print -quit 2>/dev/null)" ]]
}

if has_edge_snapshot "$EDGE_BASE_PATH"; then
    MODEL_SOURCE=$EDGE_BASE_PATH
else
    printf 'ERROR: no complete Cosmos3-Edge HF snapshot found.\n' >&2
    printf 'Checked CFS path: %s\n' "$EDGE_BASE_PATH" >&2
    printf 'Download the pinned official snapshot before conversion.\n' >&2
    exit 2
fi

DCP_PARENT=$(dirname -- "$EDGE_DCP_PATH")
STAGING_PATH="${EDGE_DCP_PATH}.incomplete"
MANIFEST_PATH="${EDGE_DCP_PATH}.manifest.tsv"

if [[ -f "$EDGE_DCP_PATH/model/.metadata" && -f "$EDGE_DCP_PATH/model/config.json" ]]; then
    printf 'DCP is already complete: %s\n' "$EDGE_DCP_PATH"
    exit 0
fi
if [[ -e "$EDGE_DCP_PATH" ]]; then
    printf 'ERROR: refusing to overwrite incomplete destination: %s\n' "$EDGE_DCP_PATH" >&2
    exit 3
fi
if [[ -e "$STAGING_PATH" ]]; then
    printf 'ERROR: staging path already exists; inspect it before retrying: %s\n' "$STAGING_PATH" >&2
    exit 4
fi

mkdir -p "$DCP_PARENT"
cd "$REPO_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH=

printf 'Converting Edge HF snapshot: %s\n' "$MODEL_SOURCE"
printf 'Staging DCP at: %s\n' "$STAGING_PATH"
"$COSMOS_PYTHON" -m cosmos_framework.scripts.convert_model_to_dcp \
    -o "$STAGING_PATH" \
    --checkpoint-path "$MODEL_SOURCE"

if [[ ! -f "$STAGING_PATH/model/.metadata" || ! -f "$STAGING_PATH/model/config.json" ]]; then
    printf 'ERROR: converter exited without complete DCP markers: %s\n' "$STAGING_PATH" >&2
    exit 5
fi

mv "$STAGING_PATH" "$EDGE_DCP_PATH"
find "$EDGE_DCP_PATH" -type f -printf '%P\t%s\n' | LC_ALL=C sort >"$MANIFEST_PATH"

printf 'DCP ready: %s\n' "$EDGE_DCP_PATH"
printf 'Size manifest: %s\n' "$MANIFEST_PATH"
