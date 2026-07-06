---
name: cosmos3-robolab-server-client
description: Run and resume Cosmos3 policy-server plus RoboLab client evaluation jobs across DROID, UR5e, X5, and other RoboLab robot backends, including remote client access, historical launch-argument consistency, asset/render failure monitoring, and clean continuation of interrupted atomic_pick or atomic_pnp batches. Use when the user asks to start a Cosmos3 action policy server for RoboLab, run or resume RoboLab tasks over SSH, audit prior launch logs, preserve output directories, choose the right robot-specific RoboLab runner, or explain/choose evaluation arguments such as --video-mode, --num-runs, --headless, --livestream, remote host/port, seed, and output folder.
---

# Cosmos3 RoboLab Server Client

## Core Rule

Preserve historical launch semantics unless the user explicitly changes them. Before starting or resuming a RoboLab batch, read current environment setup, inspect previous launch logs, and surface any behavior-changing mismatch.

Do not silently change these fields:

- `--video-mode`: default to historical value; if absent, default to `all`. Do not use `none` unless the user explicitly requests no videos.
- `--num-runs`: experiment-specific. Use the user's requested value or the historical value for that run; ask if unclear.
- `--seed`: preserve the historical/user seed when present.
- task roots, output root, randomization flags, remote host/port, and checkpoint path.
- timestamped folder/file names: use Beijing time semantics when a timestamp is needed because server clocks are often wrong, but keep the name clean; do not inject `_bjt_` or similar timezone tags into names.

## Environment Checklist

1. Read `~/.bashrc` before doing anything. Confirm `COSMOS3_FRAMEWORK_HOME`, `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, and any `LD_LIBRARY_PATH` behavior.
2. Use the RoboLab client through SSH when needed:
   ```bash
   ssh -i ~/.ssh/cosmos3_robolab_20260627 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 2222 root@10.174.136.228
   ```
3. On the client, run RoboLab Python through:
   ```bash
   /workspace/isaaclab/isaaclab.sh -p
   ```
4. On the server, validate CUDA before launching the policy server. On the 2026-06-30 mirror server, `LD_LIBRARY_PATH=/usr/local/cuda/compat/lib` was required for `torch.cuda.is_available()` to be true; clearing `LD_LIBRARY_PATH` made CUDA unavailable. Re-check on every new server.

## Server Start

Launch the Cosmos3 RoboLab action policy server from `/root/code/cosmos-framework`, using the active Cosmos3 environment and the policy DCP checkpoint. Template:

```bash
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib \
python -u -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path /mlp_vepfs/share/swy/cosmos3-framework/checkpoints/Cosmos3-Nano-Policy-DROID-dcp/model \
  --config-file /mlp_vepfs/share/swy/cosmos3-framework/checkpoints/Cosmos3-Nano-Policy-DROID-dcp/model/config.json \
  --allow-dcp-checkpoint \
  --host 0.0.0.0 \
  --port 8000 \
  --output-dir /mlp_vepfs/share/swy/cosmos3-framework/outputs/action_server_robolab_<run-id> \
  --seed <seed>
```

Verify readiness before starting the client:

```bash
curl http://<server-ip>:8000/healthz
```

Use the server IP that is reachable from the RoboLab client, not `127.0.0.1`.


### Decoupled Robot/Action Server

For new RoboLab robot backends and non-DROID action layouts, prefer the decoupled entry point:

```bash
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib \
python -u -m cosmos_framework.scripts.action_policy_server_robolab_div \
  --checkpoint-path <checkpoint>/model \
  --config-file <checkpoint-or-run>/config.yaml \
  --allow-dcp-checkpoint \
  --host 0.0.0.0 \
  --port 8000 \
  --domain-name <domain-name> \
  --action-space <joint_pos|eef_pose> \
  --action-dim <raw-model-action-dim> \
  --conditioning-fps <fps> \
  --action-chunk-size <horizon> \
  --resolution 480 \
  --output-dir /mlp_vepfs/share/swy/cosmos3-framework/outputs/action_server_robolab_<run-id> \
  --seed <seed>
```

Use `--action-space joint_pos` for direct joint-position policy output. The joint width is controlled by `--joint-dof` plus `--gripper-dim`; `--action-dim` must match that sum when provided. Use `--action-space eef_pose` for a single-arm EEF policy whose model output is 10D `[pos_delta(3), rot6d_delta(6), gripper(1)]` and whose server response is converted to the client 8D absolute EEF contract `[position(3), quat_xyzw(4), gripper(1)]`. The current EEF execution path is single-arm only; dual-arm widths are reserved but not implemented.

For checkpoints trained without a prepended state/history action row, pass:

```bash
--no-use-state --history-length 0
```

Do not mix `--use-state` / nonzero `--history-length` with a checkpoint whose dataset produced only `chunk_length` action rows.

## Client Runner Selection

Choose the client runner by robot backend before composing the command. Do not assume `policies/cosmos3/run.py` works for every robot.

| Robot/backend | Client runner | Notes |
| --- | --- | --- |
| DROID/default | `policies/cosmos3/run.py` | Uses DROID joint-position registrations and the base `Cosmos3Client`. |
| UR5e | `policies/cosmos3/run_ur5.py` | Uses UR5e joint-position registrations and `Cosmos3UR5Client`. For `--server-action-format joint`, it maps returned joint actions to the 6D UR5 arm. For `--server-action-format eef_pose`, it sends `ee_pos`/`ee_quat` and converts server 8D absolute EEF pose chunks to 6D joint actions through the Pinocchio bridge. |
| ARX X5 | `policies/cosmos3/run_x5.py` | Uses X5 joint-position registrations and the base `Cosmos3Client` with X5 camera presets. |
| Unknown/new robot | Inspect `policies/cosmos3/run_<robot>.py`, its registration import, camera presets, proprio observation keys, and action manager width before launching. Preserve the user-provided runner if they already supplied one. Match server `--action-space`, `--joint-dof`, `--gripper-dim`, and `--action-dim` to that runner. |

For explicit task files, prefer full client-side task paths such as `robolab/tasks/atomic_pnp/banana_in_bowl_task.py` over a bare filename. This avoids ambiguity between `atomic_pnp`, `benchmark`, and other task roots and ensures registration happens before the policy client is constructed.

## Client Command Template

Run each RoboLab task from `/root/code/RoboLab` on the client:

```bash
/workspace/isaaclab/isaaclab.sh -p policies/cosmos3/run.py \
  --task robolab/tasks/atomic_pnp/<task>.py \
  --remote-host <server-ip> \
  --remote-port 8000 \
  --headless \
  --video-mode all \
  --num-runs <experiment-runs> \
  --output-folder-name <run-name>/<task-slug>/<trial> \
  <trial-randomization-flags>
```

For atomic pick tasks, use `robolab/tasks/atomic_pick/<task>.py`; for pick-and-place tasks, use `robolab/tasks/atomic_pnp/<task>.py`.

UR5e single-task smoke command verified on 2026-07-04:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p policies/cosmos3/run_ur5.py \
  --task robolab/tasks/atomic_pnp/banana_in_bowl_task.py \
  --num-envs 1 \
  --num-runs 1 \
  --video-mode all \
  --headless \
  --remote-host <server-ip> \
  --remote-port 8000
```


UR5e Berkeley EEF single-task smoke template:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p policies/cosmos3/run_ur5.py \
  --task robolab/tasks/atomic_pnp/banana_in_bowl_task.py \
  --remote-host <server-ip> \
  --remote-port 8000 \
  --server-action-format eef_pose \
  --camera-preset berkeley_eef \
  --num-envs 1 \
  --num-runs 1 \
  --video-mode all \
  --headless \
  --output-folder-name <run-name>
```

The `berkeley_eef` camera preset is intended to send the Berkeley-style external + wrist + zero/missing-view canvas to the policy server. It is valid for exercising the server-client action chain, but verify wrist-camera registration and gripper execution separately before using success rate as a policy-quality metric.

Use `--video-mode all` when video capture is desired; this writes mp4 files under `/root/code/RoboLab/output/<run>/...`. Use `--video-mode none` only when the user explicitly requests no videos. Add `--livestream 2` only when the user explicitly requests livestreaming; omit it for ordinary headless recording runs.

For a full `atomic_pick` + `atomic_pnp` sweep with baseline, contact-pose randomization, and lighting randomization, prefer a batch helper that writes `manifest.tsv` and `batch_status.tsv` and runs each row as `--num-runs 1`:

```bash
/workspace/isaaclab/isaaclab.sh -p <tools>/run_atomic_batch.py \
  --run-name <run-name> --remote-host <server-ip> --remote-port 8000
```

The helper should generate 3 conditions per task: `baseline`, `pose_xy40_yawpi2` with `--randomize-contact-pose --contact-pose-xy-range 0.4 --contact-pose-yaw-range 1.5707963267948966`, and `lighting_random` through a runner wrapper that exposes `--lighting-profile random`, `--lighting-seed`, and `--env-seed`.

## Argument Semantics

| Argument | Meaning | Default/Policy |
| --- | --- | --- |
| `--task` | RoboLab task Python file on the client. | Traverse only user-requested roots, commonly `atomic_pick` and `atomic_pnp`. |
| `--remote-host` | Cosmos3 policy server IP reachable from client. | Detect current server IP; confirm client can connect. |
| `--remote-port` | Policy server websocket/HTTP port. | Usually `8000`; keep historical value if different. |
| `--headless` | Run Isaac Sim without GUI. | Keep for server/client batch runs. It is compatible with video capture. |
| `--video-mode` | Video save policy in RoboLab runner. | Use `all` by default. `none` disables mp4 output because runner sets `save_videos = args.video_mode != "none"`. Only use `none` after explicit user approval. |
| `--livestream` | IsaacLab livestream mode. | Do not add by default. It is independent from video recording; `--video-mode all` records mp4s without livestream. |
| `--num-runs` | Number of episodes per task/trial. | Experiment-specific. Do not infer from unrelated runs; use user/historical value or ask. |
| `--output-folder-name` | Output folder relative to RoboLab output root. | Keep the same run root when resuming. Do not switch roots unless requested. |
| `--randomize-contact-pose` and offsets | Trial perturbation flags. | Preserve exact values from manifest/history using structured parsing such as `shlex.split`. |
| `--seed` | Reproducibility seed when supported by the invoked script. | Preserve historical/user seed; ask if absent and reproducibility matters. |
| server `--action-space` | Server-side raw action representation. | Use `joint_pos` for joint-position policy output and `eef_pose` for EEF-pose policy output. Preserve the checkpoint's training representation. |
| server `--action-dim` | Raw model action width before server/client postprocessing. | Must match the checkpoint target width. Berkeley UR5 EEF uses 10D model output; single-arm EEF client contract after server conversion is 8D. |
| server `--joint-dof`, `--gripper-dim` | Joint-space width components for decoupled robot backends. | For `joint_pos`, `action_dim == joint_dof + gripper_dim` when `--action-dim` is provided. Current gripper execution commonly supports one gripper scalar or diagnostics-only, depending on robot. |
| server `--use-state` / `--no-use-state` | Whether the first action row is current state conditioning. | Must match training. Berkeley UR5 EEF full-run checkpoints used no state row in deployment tests, so launch with `--no-use-state --history-length 0`. |
| server `--history-length` | Number of state/history rows trimmed from generated actions. | Use `0` unless the training dataset and server request both intentionally include history action rows. |
| client `--server-action-format` | UR5 client interpretation of server response. | Use `eef_pose` when server returns `[position, quat_xyzw, gripper]`; use `joint` for joint action responses; `auto` is shape-based and should only be used when both sides are known. |
| client `--camera-preset` | UR5 camera bundle used by the runner registration. | Use `berkeley_eef` for Berkeley UR5 EEF tests; verify the physical/sim camera registration if task success is being judged. |

RoboLab code references verified on the client: `robolab/eval/runner.py:136` defines `--video-mode`; `robolab/eval/runner.py:273` sets `save_videos = args.video_mode != "none"`; `robolab/eval/runner.py:337` passes `save_videos`; `robolab/eval/runner.py:338` passes `video_mode`.


## Berkeley UR5 EEF Minimal Test

Use this paired launch pattern for a one-episode, no-randomization banana-in-bowl chain test with the Berkeley UR5 EEF checkpoint:

Server:

```bash
LD_LIBRARY_PATH=/usr/local/cuda/compat/lib \
python -u -m cosmos_framework.scripts.action_policy_server_robolab_div \
  --checkpoint-path /mlp_vepfs/share/swy/cosmos3-framework/outputs/berkeley_ur5_eef_h20/cosmos3_action/action_sft/berkeley_ur5_eef_full3000_bs8_001/checkpoints/iter_000003000/model \
  --config-file /mlp_vepfs/share/swy/cosmos3-framework/outputs/berkeley_ur5_eef_h20/cosmos3_action/action_sft/berkeley_ur5_eef_full3000_bs8_001/config.yaml \
  --allow-dcp-checkpoint \
  --host 0.0.0.0 \
  --port 8000 \
  --domain-name berkeley-ur5-eef \
  --action-space eef_pose \
  --action-dim 10 \
  --conditioning-fps 5 \
  --action-chunk-size 32 \
  --resolution 480 \
  --no-use-state \
  --history-length 0 \
  --seed 0 \
  --deterministic-seed \
  --output-dir /mlp_vepfs/share/swy/cosmos3-framework/outputs/action_server_robolab_berkeley_ur5_eef_<run-id>
```

Client:

```bash
ssh -i ~/.ssh/cosmos3_robolab_20260627 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 2222 root@10.174.136.228 \
  'cd /root/code/RoboLab && TERM=xterm /workspace/isaaclab/isaaclab.sh -p policies/cosmos3/run_ur5.py \
    --task robolab/tasks/atomic_pnp/banana_in_bowl_task.py \
    --remote-host <server-ip> \
    --remote-port 8000 \
    --server-action-format eef_pose \
    --camera-preset berkeley_eef \
    --num-envs 1 \
    --num-runs 1 \
    --video-mode all \
    --headless \
    --output-folder-name <run-id>'
```

Expected output location on the client:

```bash
/root/code/RoboLab/output/<run-id>/
```

This test validates the current server-client contract and launch parameters. It does not by itself validate Berkeley/RoboLab camera alignment, wrist-camera registration quality, or gripper-controlled task success.

## Resume Workflow

1. Identify the output root and batch log directory, e.g. `/root/code/RoboLab/output/<run-name>/_batch_logs`.
2. Inspect prior launch commands from `_batch_logs/*.log`, `resume_outer_*.log`, launcher logs, manifests, and status TSVs.
3. Build a manifest of task/trial rows. Mark rows with `rc=0`, `asset_errors=0`, and `status=ok` as complete.
4. Do not rerun rows already marked `ok` unless the user explicitly asks for regeneration.
5. Preserve output root and append status to the same run, archiving partial/incomplete outputs before rerun when necessary.
6. Record every command, return code, asset/render hit count, duration, and status in `batch_status.tsv`.
7. After completion, report counts: total rows, ok, skipped asset/render errors, generic errors, and severe hit count.

## Asset And Render Guard

Continuously scan the active client task log and batch logs. Treat the following as serious unless clearly whitelisted by prior project knowledge:

```text
[Error].*(References an asset that can not be found|Couldn't process file|Failed to load image|STB Failed|asset|texture|usd|render)
Terminated with error
Traceback
Segmentation fault
core dumped
Could not open|Failed to open|Failed to load|Unable to open|Could not load
```

Do not classify these common Isaac/RoboLab warnings as fatal by themselves:

```text
GLFW initialization failed in headless mode
Failed to startup plugin carb.windowing-glfw.plugin
failed to open the default display
Source: omni.hydra was already registered
omni.isaac.dynamic_control is deprecated
No material configuration file
Parameter .* not available in the MDL representation
DLSS increasing input dimensions
using high frequency span with attrs is disabled
ImplicitActuatorCfg effort_limit/velocity_limit warnings
Not all actuators are configured
```

If a serious asset/render/import failure appears:

1. Stop rerunning further trials for the same task unless the user says otherwise.
2. Mark the current row as `asset_render_error` and later trials of that task as `skipped_asset_render_error`.
3. Continue other tasks.
4. Report the log path, matching line, task, trial, and exact status decision.

## Historical Consistency Checks

Before launch, compare intended args against the historical command. If any of these differ, ask before running:

- `--video-mode`, especially changing `all` to `none`.
- `--num-runs`, unless the user already specified the new experiment requirement.
- output root or task root.
- timestamp source or timestamped folder/file naming convention.
- checkpoint path, config file, `--allow-dcp-checkpoint`, host/port, and seed.
- randomization flags or contact-pose offsets.

When the user provides a historical fragment such as:

```bash
--headless \
--video-mode all \
--num-runs 1
```

preserve `--video-mode all`; treat `--num-runs 1` as the historical experiment setting and only change it when the user says the new experiment requires a different run count.

## Useful Verification Commands

Known-good full sweep: `atomic_pick_pnp_20260630_215055` under `/mlp_vepfs/share/swy/cosmos3-framework/client_outputs` completed `rows=270 ok=270 asset_or_skip=0 error=0 timeout=0 resource_skip=0` with 540 mp4s. The only issue was a false-positive guard match on headless display startup warnings; whitelist those, but keep real asset/import/render errors fatal.

Client status snapshot:

```bash
ssh -i ~/.ssh/cosmos3_robolab_20260627 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 2222 root@10.174.136.228 \
  'cd /root/code/RoboLab && RUN=output/<run-name> && tail -8 "$RUN/batch_status.tsv" && find "$RUN" -name "*.mp4" | wc -l'
```

Final consistency checks:

```bash
awk -F'\t' 'NR>1{n++; if($4==0 && $5==0 && $6=="ok") ok++; if($6 ~ /asset_render_error|skipped_asset_render_error/) bad++; if($6=="error") err++} END{printf "rows=%d ok=%d asset_or_skip=%d error=%d\n", n+0, ok+0, bad+0, err+0}' batch_status.tsv
```

Server process check:

```bash
ps -p <server-pid> -o pid,stat,etime,cmd
```
