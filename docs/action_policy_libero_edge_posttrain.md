# Cosmos3-Edge LIBERO-all action-policy SFT

This recipe post-trains the public `nvidia/Cosmos3-Edge` checkpoint on an equal
mixture of the four standard LIBERO suites. It is a first-principles Edge port
of the repository's Nano LIBERO-all baseline: the data representation,
optimizer, learning-rate schedule, 5,000 optimizer steps, and effective global
batch remain unchanged; the backbone/configuration is replaced by the native
Nemotron-2B-Dense-VL Edge stack.

The two launch sizes are deliberately optimization-equivalent:

| Preset | FSDP shard | Replicate | Context parallel | Per-rank samples | Grad accumulation | Effective global batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 GPU | 8 | 1 | 1 | 128 | 2 | 2048 |
| 4 GPU | 4 | 1 | 1 | 128 | 4 | 2048 |

The effective batch is `samples/rank × world size × grad accumulation`. The
read-only RPBZZZ6 runtime logs visible on this development machine report
97,887 MB per GPU, so the initial recipe keeps the established 128-sample
per-rank LIBERO cap. Four GPUs hold larger FSDP parameter/optimizer shards and
perform twice as many accumulation passes; expect it to be roughly twice as
slow per optimizer step, subject to input and communication throughput.

These Edge hyperparameters are an informed port, not an already converged Edge
golden. Run the smoke sequence below before committing the full allocation, and
track action loss, vision loss, grad norm, throughput, and peak reserved memory.

## Files

| Component | Path |
| --- | --- |
| Shared experiment | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_libero_all_edge.py` |
| 8-GPU TOML | `examples/toml/sft_config/action_policy_libero_all_edge_8gpu.toml` |
| 4-GPU TOML | `examples/toml/sft_config/action_policy_libero_all_edge_4gpu.toml` |
| 8-GPU launcher | `examples/launch_sft_action_policy_libero_all_edge_8gpu.sh` |
| 4-GPU launcher | `examples/launch_sft_action_policy_libero_all_edge_4gpu.sh` |

Both presets use 20 FPS observations, 16-step action chunks, `concat_view`,
256×256 images per camera, frame-wise-relative rot6d actions, and
`quantile_rot` normalization. `LIBERO_ROOT` is the parent of the four suite
directories, not a single suite.

## Prepare assets

Download the public Edge snapshot and convert it to DCP once:

```bash
hf download nvidia/Cosmos3-Edge \
  --revision 6f58f6b4c91288838e60b6bcb2cc45d997e961de \
  --local-dir /path/to/assets/Cosmos3-Edge

python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o /path/to/checkpoints/Cosmos3-Edge \
  --checkpoint-path /path/to/assets/Cosmos3-Edge
```

Download the four-suite training dataset:

```bash
hf download nvidia/LIBERO_LeRobot_v3 \
  --repo-type dataset \
  --revision ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64 \
  --local-dir /path/to/datasets/LIBERO_LeRobot_v3
```

Export the runtime paths. The Wan2.2 VAE may remain on a read-only shared
mount; checkpoints, logs, caches, and outputs must be writable.

```bash
export LD_LIBRARY_PATH=''
export LIBERO_ROOT=/path/to/datasets/LIBERO_LeRobot_v3
export BASE_CHECKPOINT_PATH=/path/to/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=/mnt/bos/1011/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs/cosmos3-edge-libero
```

## Preflight and launch

First validate the assembled environment, then run a short no-W&B smoke. The
smoke still writes ordinary training output under `IMAGINAIRE_OUTPUT_ROOT`, but
does not save a checkpoint in five iterations.

```bash
bash edge_task/verify_environment.sh

EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 checkpoint.save_iter=999999 job.wandb_mode=disabled" \
  bash examples/launch_sft_action_policy_libero_all_edge_8gpu.sh
```

For the full run, choose exactly one launcher matching the allocated GPU count:

```bash
# One node, 8 GPUs
bash examples/launch_sft_action_policy_libero_all_edge_8gpu.sh

# One node, 4 GPUs
bash examples/launch_sft_action_policy_libero_all_edge_4gpu.sh
```

The TOMLs default to `wandb_mode="online"`; provide `WANDB_API_KEY`, or launch
with `EXTRA_TAIL_OVERRIDES="job.wandb_mode=offline"` when external logging is
not configured.

The launchers set `NPROC_PER_NODE` themselves; they do not depend on changing
the container's `docker run` command. They fail early if any suite, the Edge
DCP, or the VAE is missing.

## Memory fallback

If the first real batch OOMs, halve the per-rank cap and double gradient
accumulation so the effective global batch and learning-rate assumptions stay
unchanged:

```bash
# 8 GPUs: 64 × 8 × 4 = 2048
EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=4" \
  bash examples/launch_sft_action_policy_libero_all_edge_8gpu.sh

# 4 GPUs: 64 × 4 × 8 = 2048
EXTRA_TAIL_OVERRIDES="dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=8" \
  bash examples/launch_sft_action_policy_libero_all_edge_4gpu.sh
```

Do not change `data_parallel_replicate_degree` on these single-node presets:
the shard degree already equals `WORLD_SIZE`. Context and CFG parallelism stay
at one because the 74k packed-token cap and selective activation checkpointing
are the intended memory controls for this Edge action recipe.
