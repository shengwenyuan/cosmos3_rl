#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/cosmos3_rl

run=/mnt/pfs/swy/cosmos3/runs/cosmos3_action_rh20t_ur5_joint/cosmos3_action_rh20t_ur5_joint/stage2_c32/rh20t_cfg4_ur5_joint_edge_64gpu_10k
python_bin=${PYTHON_BIN:-/mnt/pfs/swy/cosmos3/envs/cosmos-framework-cu130-train/bin/python}
seed=${SEED:-0}
output=${OUTPUT:-$run/evaluation/iter_000010000/gt_chunk_audit_seed$seed}
capture=$output/capture.json

if [[ ! -x "$python_bin" ]]; then
  echo "python is not executable: $python_bin" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 2
fi
mkdir -p "$output"
env LD_LIBRARY_PATH='' "$python_bin" tools/rh20t/evaluate_cfg4_ur5_joint_policy.py \
  --run "$run" --iteration 10000 --samples 32 --seed "$seed" \
  --num-steps 30 --guidance 1 --action-normalization-depth 2 \
  --joint-chunk-postprocessor triangular_3tap --output "$capture"

env LD_LIBRARY_PATH='' "$python_bin" -m tools.rh20t.gt_chunk_audit.analyze \
  --samples "${capture%.json}_samples.npz" --report "$capture" --output "$output/chunks"
