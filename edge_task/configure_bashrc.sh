#!/usr/bin/env bash
set -euo pipefail

BASHRC_FILE="$HOME/.bashrc"
WANDB_CREDENTIALS_FILE="/mnt/cfs/data/swy/personal/wandb"
TEMP_FILE=$(mktemp)
trap 'rm -f "$TEMP_FILE"' EXIT

[[ -f "$WANDB_CREDENTIALS_FILE" && ! -L "$WANDB_CREDENTIALS_FILE" && -r "$WANDB_CREDENTIALS_FILE" ]] || {
    printf 'Missing readable W&B credentials file: %s\n' "$WANDB_CREDENTIALS_FILE" >&2
    exit 1
}
chmod 600 "$WANDB_CREDENTIALS_FILE"
WANDB_ENTITY_VALUE=$(sed -n '1{s/\r$//;p;}' "$WANDB_CREDENTIALS_FILE")
WANDB_API_KEY_VALUE=$(sed -n '2{s/\r$//;p;}' "$WANDB_CREDENTIALS_FILE")
LINE_COUNT=$(awk 'END { print NR }' "$WANDB_CREDENTIALS_FILE")
[[ "$WANDB_ENTITY_VALUE" == "sals-northeastern-university" && -n "$WANDB_API_KEY_VALUE" && "$LINE_COUNT" -eq 2 ]] || {
    printf 'Invalid W&B credentials format in %s\n' "$WANDB_CREDENTIALS_FILE" >&2
    exit 1
}
unset WANDB_ENTITY_VALUE WANDB_API_KEY_VALUE LINE_COUNT
mkdir -p /mnt/cfs/data/swy/cosmos3/runs /mnt/cfs/data/swy/cosmos3/cache/wandb /mnt/cfs/data/swy/cosmos3/cache/wandb-data
touch "$BASHRC_FILE"

{
    printf '%s\n' \
        '# >>> cosmos-edge-libero >>>' \
        'source /root/workspace/cosmos-framework/edge_task/env.example.sh' \
        'export WANDB_ENTITY="$(sed -n '\''1{s/\r$//;p;}'\'' /mnt/cfs/data/swy/personal/wandb)"' \
        'export WANDB_API_KEY="$(sed -n '\''2{s/\r$//;p;}'\'' /mnt/cfs/data/swy/personal/wandb)"' \
        '# <<< cosmos-edge-libero <<<'
    sed '/^# >>> cosmos-edge-libero >>>$/,/^# <<< cosmos-edge-libero <<<$/{d;}' "$BASHRC_FILE"
} >"$TEMP_FILE"
cat "$TEMP_FILE" >"$BASHRC_FILE"

printf 'Configured %s with W&B credentials from %s\n' "$BASHRC_FILE" "$WANDB_CREDENTIALS_FILE"
