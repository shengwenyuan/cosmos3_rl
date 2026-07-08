---
name: cosmos3-volcengine-training
description: Guide Cosmos3 Volcengine/MLP custom training jobs from preflight through launch, W&B supervision, failure triage, and final analysis. Use when preparing, reviewing, configuring, launching, monitoring, or analyzing a Volcengine custom training task, especially scripts like examples/launch_volcengine_berkeley_ur5_eef_h20.sh.
---

# Cosmos3 Volcengine Training

## Core Rule

Treat a Volcengine custom training task as a staged production run. Preserve the validated launcher defaults, override frequently changed values through environment variables, and write a short run document before launch plus a short analysis document after completion.

Do not jump directly from an edited image or script to a long full run. Use the ladder: preflight -> dryrun -> smoke/probe -> full run.

## When To Use This Skill

Use this skill when the user asks to:

- prepare or review a Volcengine custom training task.
- configure the Volcengine UI entry command and environment variables.
- inspect or extend `../../../examples/launch_volcengine_*.sh`.
- decide preflight, dryrun, smoke, pilot, or full-training settings.
- plan W&B monitoring and training stop criteria.
- triage a failed cloud run from stdout, log files, or W&B.
- write the pre-launch report or post-run analysis under `../../../tmps/`.
- draft a future Volcengine launcher for another Cosmos3 training recipe.

## Path Convention

All paths in this skill are relative to the repository root unless explicitly absolute.

Before acting on a Volcengine training request, read `~/.bashrc` for the active shared roots, venv, token locations, and CUDA library handling. In the current deployment pattern:

- `/mlp_vepfs/share/swy/cosmos3-framework` is the fast shared workspace for checkpoints, datasets, venvs, outputs, and logs.
- `/dexmal-datainfra-swy/bootstrap` is the slow personal bootstrap area for init snippets and private tokens.
- `LD_LIBRARY_PATH` is expected to be cleared before local Python imports unless the Volcengine CUDA forward-compat path is intentionally enabled.

## Volcengine Task Shape

For CLI-submitted tasks, also read `references/volc_cli_training.md` before
drafting the YAML or final `volc ml_task submit` command.

Use one shell launcher as the task entry command, for example:

```bash
bash examples/launch_volcengine_berkeley_ur5_eef_h20.sh
```

Volcengine UI environment variables are plain key/value strings. Do not paste shell syntax such as `export FOO=...`, quotes that become part of the value, or placeholder values like `<rank-0 host or IP>`.

For a single 8-H20 node, the usual topology is:

```text
NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
DP_SHARD=8
MAX_SAMPLES_PER_BATCH=auto or 8
USE_CUDA_COMPAT=required
MLP_LOG_PATH=/root/logs
WANDB_MODE=online
```

For multi-node jobs, explicitly set `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`, or confirm that the platform-provided `MLP_*`/`VC_*` variables are mapped by the launcher. Do not assume the UI fills placeholder text for you.

Use `MLP_LOG_PATH=/root/logs` so Volcengine can collect `.log` files outside stdout. Keep W&B credentials out of the UI when possible; prefer a token file such as `/dexmal-datainfra-swy/bootstrap/wandb_token`.

## Current Berkeley EEF Launcher Map

The current Berkeley UR5 EEF cloud entry point is:

- `../../../examples/launch_volcengine_berkeley_ur5_eef_h20.sh`
- It is an outer Volcengine wrapper and delegates actual training to `../../../examples/launch_sft_action_policy_berkeley_ur5_eef.sh`.
- It expects the Berkeley LeRobot dataset, a DCP base checkpoint, and a Wan2.2 VAE path from environment variables or defaults.
- It supports `PREFLIGHT_ONLY`, `DRYRUN_ONLY`, `RUN_DRYRUN_FIRST`, `TOPOLOGY_ONLY`, `WANDB_MODE`, `USE_CUDA_COMPAT`, `MAX_ITER`, `SAVE_ITER`, `LOGGING_ITER`, `MAX_SAMPLES_PER_BATCH`, and `EXTRA_TAIL_OVERRIDES`.
- It validates path existence, dataset metadata, hardware visibility, W&B setup, topology, and placeholder-looking values before launching.
- It writes orchestration logs under the shared output root and mirrors them to `$MLP_LOG_PATH` when that variable is set.
- It builds command-line tail overrides for `job.name`, `job.wandb_mode`, trainer iterations, checkpoint cadence, dataloader batch, and model/data parallelism.
- It runs an optional direct Python dryrun before invoking the inner `torchrun` launcher.

When reviewing this script, inspect these areas first:

- defaults and environment knobs near the top of the file.
- `validate_static_env_values` for UI placeholder protection.
- `resolve_topology` for node/rank/world-size behavior.
- `setup_logging` for shared and Volcengine-collected logs.
- `validate_paths`, `validate_dataset_metadata`, `validate_hardware`, and `validate_wandb`.
- `build_tail_overrides` for the final training config passed to Cosmos3.
- `run_dryrun` and `launch_training` for the actual execution path.

## Pre-launch Report Checklist

Before a non-trivial cloud run, write or update `../../../tmps/<job_name>_training_report.md`.

Keep it concise and include:

- Task goal: what this run is meant to prove or produce.
- Dataset: root path, format, expected episode/frame counts, cameras, action semantics.
- Checkpoints: base/resume checkpoint, VAE path, output root, checkpoint cadence.
- Launch command: the exact Volcengine entry command.
- Environment table: all UI variables that differ from launcher defaults.
- Topology: nodes, GPUs per node, world size, `DP_SHARD`, expected H20 memory.
- Hyperparameters: `MAX_ITER`, batch, LR overrides, scheduler overrides, logging cadence, save cadence.
- Expected behavior: approximate speed, expected checkpoints, W&B run/project, loss trend expectation.
- Known risks: unvalidated coordinate frames, batch-size memory margin, action-head initialization, missing validation/eval loop, resume semantics.
- Stop criteria: NaN, repeated OOM, no checkpoints, W&B step stalls, loss explosion, data loader failure.

Use prior run reports in `../../../tmps/` as style references, but do not copy stale hyperparameters without checking the launcher and current user request.

## Launch Ladder

Use this escalation order after any meaningful image, script, environment, or dataset change:

1. `PREFLIGHT_ONLY=1`: environment, paths, hardware, CUDA compat, W&B token, topology.
2. `DRYRUN_ONLY=1` or `RUN_DRYRUN_FIRST=1`: Python config validation without distributed training.
3. Smoke/probe: 10-120 iterations, frequent logging, checkpoint at the end.
4. Pilot: hundreds of iterations only if the smoke run proves speed, memory, logging, and checkpointing.
5. Full run: longer schedule with the intended batch, checkpoint cadence, and W&B mode.

If a stage fails, do not only increase runtime and retry. Identify whether the failure is resource-side, launcher-side, data-side, config-side, or training-side.

## Training Supervision

Plan W&B monitoring before launch. Track at minimum:

- Run state and heartbeat: running, finished, failed, crashed, or stalled.
- Step progress: iteration and sample counter.
- Loss: `train/loss` or equivalent aggregate loss, first/latest/best/window averages.
- Component losses when present: action, vision, decoder, or task-specific details.
- Optimization: learning rate, grad norm, skipped/overflow steps if logged.
- Throughput: seconds per iteration, samples per second, video batch size, packing/padding stats.
- Hardware: GPU memory, utilization, MFU/OFU, dataloader stalls if callbacks expose them.
- Checkpoints: latest checkpoint path, save cadence, and whether the last expected checkpoint exists.

Also monitor logs:

- shared orchestration logs under the configured output root.
- shared training logs under the configured output root.
- Volcengine-collected logs under `$MLP_LOG_PATH`.
- stdout only as a fallback; stdout is often truncated and may omit useful rank-local details.

If local W&B helpers exist, prefer them for repeatable snapshots and append-only analysis files. Otherwise use the W&B API or manual screenshots plus exact timestamps.

## End-of-run Analysis

After completion or failure, update `../../../tmps/<run_or_monitor>/analysis.md` or create a similarly named run analysis.

Include:

- Final state: finished, failed, killed, or manually stopped.
- Final iteration/sample counter and wall-clock duration.
- Checkpoints produced and the recommended checkpoint to evaluate or resume from.
- Loss summary: first, latest, best, last-window mean, and whether the tail still improves.
- Component-loss behavior and any mismatch between aggregate loss and action-specific loss.
- LR/scheduler state at the end.
- Throughput stability and hardware saturation.
- Failure root cause if applicable, with log line references.
- Fitting judgment: undertrained, still improving, plateaued, overfit risk, or inconclusive.
- Validation gap: whether this run only proves supervised loss, and what RoboLab/simulation/real-robot evaluation is still required.
- Next run recommendation: continue, increase/decrease iterations, change batch, change LR, fix data semantics, or move to deployment validation.

For action policies, supervised training loss is not a complete validation loop. Treat embodiment evaluation, RoboLab tasks, or held-out success metrics as the real policy-quality check when available.

## Writing Future Volcengine Launchers

Use the current Berkeley launcher as the template, but keep dataset-specific assumptions isolated.

Required launcher structure:

- `set -Eeuo pipefail` and an error trap that prints shared and `$MLP_LOG_PATH` log locations.
- top-of-file environment defaults with user-overridable variables.
- bootstrap sourcing from `/dexmal-datainfra-swy/bootstrap` when present.
- venv activation and explicit CUDA library handling.
- topology resolution from UI variables and platform variables.
- path validation for dataset, checkpoint, VAE, TOML, and inner launcher.
- hardware and driver checks, including optional CUDA forward compatibility.
- W&B validation that supports online, offline, and disabled modes.
- metadata emission for reproducibility.
- `PREFLIGHT_ONLY`, `DRYRUN_ONLY`, `RUN_DRYRUN_FIRST`, and `TOPOLOGY_ONLY`.
- tail overrides instead of editing TOML for routine run changes.
- shared logs plus `$MLP_LOG_PATH` logs for Volcengine downloadability.

Avoid:

- hardcoded secrets.
- hardcoded per-run hyperparameters that should be UI environment variables.
- silent fallback from a missing dataset/checkpoint to another path.
- accepting placeholder UI values.
- increasing batch size beyond proven memory margin without an explicit risk flag.
- resuming from a checkpoint in a way that accidentally skips freshly initialized task heads.

## Related Skills

- `cosmos3-post-training`: local SFT workflow, DCP checkpoints, TOML recipes, and training entry points.
- `cosmos3-env-troubleshoot`: CUDA, import, dependency, and runtime failure triage.
- `cosmos3-codebase-nav`: locating configs, defaults, callbacks, and data loaders.
- `cosmos3-robolab-server-client`: deployment/evaluation loops after a trained action policy is available.
