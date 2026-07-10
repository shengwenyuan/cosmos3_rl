# RoboMIND1 UR5 Single-Top-View Joint Post-Training Plan

Status: metadata-only preflight passed for the
`/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint`
LeRobot dataset. This is a run-specific supplement to
`docs/action_policy_robomind_ur5_posttrain.md`.

## Target Contract

| item | decision |
| --- | --- |
| dataset root | `/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint` |
| robot/action | UR5 joint-space policy, 6 arm joints + 1 gripper = 7D |
| visual layout | DROID-style fixed three-view canvas: real top third-person camera on top, two masked/black missing-view placeholders on bottom |
| gripper convention | uncertain; first run keeps RoboMIND gripper raw / no training flip, with RoboLab client unchanged |
| experiment | `action_policy_robomind_ur5_single_nano` |
| serving entry point | use `action_policy_server_robolab_div.py`; `action_policy_server_robolab.py` is not usable unchanged for 6+1 UR5 |

Observed dataset facts:

- `meta/info.json` is LeRobot v3, `robot_type=ur5`, `fps=15`, 22044 episodes, 2965157 frames.
- Split fields match the current adapter: `action.arm_left_joint` [6], `action.gripper_left` [1],
  `observation.state.arm_left_joint` [6], `observation.state.gripper_left` [1].
- The only real image feature is `observation.images.camera_top` with 360x640 AV1 video. For this run,
  treat that feature as a top/overhead third-person view, not a wrist camera.
- Estimated stride-1 training windows with `chunk_length=32`: `2965157 - 22044 * 32 = 2259749`.
  On 8 GPUs with `max_samples_per_batch=128`, one pass over those windows is about 2207 optimizer steps;
  a 10000-step run is roughly 4.5 passes.

## Current Chain

1. **Launcher path.** The canonical launcher sets the TOML and bridges `DATASET_PATH` into
   `UR5_SINGLE_ROOT` (`examples/launch_sft_action_policy_robomind_ur5_single.sh:24`,
   `examples/launch_sft_action_policy_robomind_ur5_single.sh:29`). Its built-in default still points at
   `/mlp_vepfs/.../RoboMIND1-ur5`, so this run must export `DATASET_PATH`/`UR5_SINGLE_ROOT` explicitly or
   update the default (`examples/launch_sft_action_policy_robomind_ur5_single.sh:25`).

2. **Local H200 wrapper.** `examples/launch_paidsw_ur5_single.sh` performs path, DCP, VAE, metadata, hardware,
   W&B, dryrun, and launch orchestration. Its dataset default is also the stale `RoboMIND1-ur5` path
   (`examples/launch_paidsw_ur5_single.sh:17`), but its metadata validator already expects the right split
   6+1 schema (`examples/launch_paidsw_ur5_single.sh:191`).

3. **Recipe selection.** The TOML selects `action_policy_robomind_ur5_single_nano`
   (`examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml:18`) and loads the DCP/VAE from
   environment variables (`examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml:36`,
   `examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml:46`).

4. **Experiment registration.** The single-arm experiment is registered with `which_arm="left"` and
   `root_env="UR5_SINGLE_ROOT"` (`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robomind_ur5_single_nano.py:25`).
   The config registry imports it, so the TOML experiment name resolves at load time
   (`cosmos_framework/configs/base/config.py:103`).

5. **Dataloader config.** The shared builder uses `chunk_length=32`, `mode="policy"`, `use_state=True`,
   `which_arm=left`, raw action normalization, and the fixed three-view zero-pad canvas
   (`cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:189`,
   `cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:190`,
   `cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:191`,
   `cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:192`,
   `cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:196`,
   `cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:200`).

6. **Action width.** `RoboMINDUR5Dataset` computes single-arm `action_dim` as `6 + 1 = 7`
   (`cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:141`) and builds a joint+gripper
   action spec (`cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:161`). The transform
   then records the real width as `raw_action_dim` before padding to `max_action_dim`
   (`cosmos_framework/data/vfm/action/action_processing.py:206`,
   `cosmos_framework/data/vfm/action/action_processing.py:215`,
   `cosmos_framework/data/vfm/action/action_processing.py:220`). This already covers the DROID 8D -> UR5 7D
   reduction; no extra text-only "dimension description" is needed for the model loss.

7. **Canvas.** With one image feature, the adapter decodes only that feature and passes two zero views into
   `concat_three_view_canvas` (`cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:222`,
   `cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:315`,
   `cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:316`). The zero helper really creates
   black views (`cosmos_framework/data/vfm/action/datasets/canvas_utils.py:12`).

8. **Caption path.** Dataset-specific camera text is passed via `additional_view_description`
   (`cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:195`), then appended by
   `ViewpointTextInfo` after the generic `concat_view` sentence
   (`cosmos_framework/data/vfm/action/viewpoint_utils.py:91`,
   `cosmos_framework/data/vfm/action/viewpoint_utils.py:102`). Current UR5 text is still generic/overview
   (`cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py:255`), so it should be changed for this
   single-top-view run.

9. **Serving.** The 6+1 UR5 checkpoint needs a 6-DoF-aware server. `action_policy_server_robolab_div.py`
   derives joint-pos width from `--joint-dof` and `--gripper-dim`
   (`cosmos_framework/scripts/action_policy_server_robolab_div.py:418`) and therefore supports UR5 as
   `--joint-dof 6 --gripper-dim 1`. It also exposes the concat-view caption as `--view-description`. Its generic
   three-camera default is not the target layout for this dataset; override it with the top-view wording below.
   The older `action_policy_server_robolab.py` is
   DROID/Franka-shaped:
   it defaults to raw action width 8 (`cosmos_framework/scripts/action_policy_server_robolab.py:348`),
   resolves joint-pos action width through that default (`cosmos_framework/scripts/action_policy_server_robolab.py:401`),
   and hard-codes observation joint width 7 before appending gripper
   (`cosmos_framework/scripts/action_policy_server_robolab.py:522`,
   `cosmos_framework/scripts/action_policy_server_robolab.py:524`). Passing UR5 6-joint observations will fail
   width validation; forcing `--action-dim 7` while still giving 7 joints would fail assignment because the
   concatenated joint+gripper state is 8D.

## Required Small Deltas Before Full Training

1. **Point launch defaults or env at the new dataset.**
   Minimum: export both vars before launch:

   ```bash
   export DATASET_PATH=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint
   export UR5_SINGLE_ROOT=$DATASET_PATH
   ```

   Cleaner: update the two stale defaults in `examples/launch_sft_action_policy_robomind_ur5_single.sh:25`
   and `examples/launch_paidsw_ur5_single.sh:17`.

2. **Keep gripper flip off for the first run.**
   Current config keeps RoboMIND gripper raw (`cosmos_framework/configs/base/experiment/action/posttrain_config/_robomind_ur5_common.py:193`).
   The official material does not settle the gripper polarity for this UR5 path. The first-run assumption is:
   do not change the RoboLab client, do not set `gripper_invert=True`, keep the server's existing boundary
   convention unless separately patched, and judge polarity from closed-loop behavior. This assumes RoboMIND raw
   gripper already matches the DROID server's model-space direction, because DROID joint training flips gripper
   with `1.0 - gripper`
   (`cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py:277`,
   `cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py:283`) and both RoboLab server variants
   currently flip gripper at the observation/output boundary
   (`cosmos_framework/scripts/action_policy_server_robolab.py:519`,
   `cosmos_framework/scripts/action_policy_server_robolab.py:591`,
   `cosmos_framework/scripts/action_policy_server_robolab_div.py:611`,
   `cosmos_framework/scripts/action_policy_server_robolab_div.py:685`). If arm trajectories look correct but grasp
   open/close is obviously inverted, rerun with a single polarity change instead of mixing client, server, and
   training-side changes.

3. **Use the top third-person camera caption.**
   Replace the one-view/three-view UR5 view description with a deterministic sentence such as:

   ```text
   The top row is a top-down third-person camera view of the UR5 workspace. The bottom row contains two masked missing camera views rendered as black images.
   ```

   The same wording should be used by the serving path when the client sends the training-matched precomposed
   top-view + black-placeholder canvas:

   ```bash
   --view-description "The top row is a top-down third-person camera view of the UR5 workspace. The bottom row contains two masked missing camera views rendered as black images."
   ```

   If the RoboLab client later sends three real views instead, update the caption to describe those real views.

4. **Serve with UR5 dimensions and the UR5 domain.**
   Expected server shape flags:

   ```bash
   python -m cosmos_framework.scripts.action_policy_server_robolab_div \
     --checkpoint-path <trained-or-exported-checkpoint> \
     --allow-dcp-checkpoint \
     --domain-name robomind-ur5-single \
     --action-space joint_pos \
     --joint-dof 6 \
     --gripper-dim 1 \
     --image-height 540 \
     --image-width 640 \
     --conditioning-fps 15 \
     --action-chunk-size 32 \
     --view-description "The top row is a top-down third-person camera view of the UR5 workspace. The bottom row contains two masked missing camera views rendered as black images."
   ```

   Directly using `cosmos_framework.scripts.action_policy_server_robolab` without edits is not a valid 6+1 path:
   it expects 7 robot joints before gripper. The only no-`_div` route that preserves this training contract is to
   port the `--joint-dof`/`--gripper-dim` behavior from `_div` into `action_policy_server_robolab.py`; padding the
   client to fake an 8D Franka-shaped action would change the model contract.

   The client should send either a precomposed `observation/image` matching the train canvas, or the server/client
   boundary must be patched to compose top-view + two black placeholders from the single top camera. Current
   `_extract_observation_image` accepts either a precomposed image or the RoBoArena wrist/exterior triplet; that
   triplet is not this dataset's visual contract (`cosmos_framework/scripts/action_policy_server_robolab_div.py:227`,
   `cosmos_framework/scripts/action_policy_server_robolab_div.py:244`).

## Validation Plan

1. **Static preflight.** Completed on 2026-07-10 with GPU visibility disabled, explicit dataset paths, and hardware
   checks skipped:

   ```bash
   CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void CUDA_DEVICE_ORDER=PCI_BUS_ID \
   DATASET_PATH=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint \
   UR5_SINGLE_ROOT=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint \
   OUTPUT_ROOT=/tmp/cosmos_ur5_single_preflight \
   IMAGINAIRE_OUTPUT_ROOT=/tmp/cosmos_ur5_single_preflight \
   WANDB_MODE=disabled SKIP_HARDWARE_CHECK=1 PREFLIGHT_ONLY=1 RUN_DRYRUN_FIRST=0 \
   bash examples/launch_paidsw_ur5_single.sh
   ```

   Result: PASS. The wrapper reported `episodes=22044`, `frames=2965157`, `fps=15`; validated raw 7D action
   contract with `use_state=True`; accepted the single real top-view `camera_top` feature with zero-padded missing views;
   disabled W&B; skipped hardware check intentionally; and exited before dryrun/training. No CUDA context or
   training process was launched.

2. **Dataset sample check after code deltas.** Instantiate `RoboMINDUR5Dataset` in the training env and verify:
   action shape is `(33, 7)`, `raw_action_dim=7` after transform, top/bottom canvas is 540x640 before resize, bottom
   two views are black, and caption contains "top-down" or "third-person" plus "masked missing".

3. **Dryrun.** Wait until the current 8-card full training releases enough memory, then run `DRYRUN_ONLY=1` through
   `examples/launch_paidsw_ur5_single.sh` with the exact dataset, checkpoint, VAE, no-training-flip gripper policy,
   caption code, and intended overrides. Before launching, cap visible GPUs explicitly so the active training
   process is not disturbed.

4. **Short smoke.** Run 10-50 iterations with `WANDB_MODE=disabled` first. Save one batch sample or decoded frame
   if possible to audit canvas and prompt text.

5. **Real run.** For 8xH200, keep the wrapper's scaled LR (`2.5e-5`) unless changing effective batch
   (`examples/launch_paidsw_ur5_single.sh:51`). If scaling to 64 ranks, recompute `MAX_ITER`; 10000 steps would be
   many passes over this 2.26M-window dataset.

6. **Closed-loop evaluation.** Start a 6-DoF-aware RoboLab server with
   `--joint-dof 6 --domain-name robomind-ur5-single` and verify one action chunk shape `(32, 7)` before running
   task batches against the RoboLab client. For the first run, leave gripper client behavior unchanged; only
   retrain/flip if arm motion is plausible and gripper polarity is the isolated failure.

## Current Risks

| risk | why it matters | mitigation |
| --- | --- | --- |
| Gripper polarity mismatch | Official material does not make the UR5 gripper convention certain. DROID uses `1-g`, but this UR5 run will first keep raw RoboMIND gripper and keep the RoboLab client unchanged. | Treat this as an experiment variable. If arm motion is correct but gripper open/close is wrong, rerun with training-side `gripper_invert=True` or an equivalent single boundary flip, not both. |
| Caption/canvas mismatch | Training target is top third-person view + two masked black missing views. The divided server default is a generic three-camera sentence, which is not correct for this run. | Pass `--view-description` matching the actual served canvas; audit transformed prompt and one server request sample. |
| Stale default dataset path | Launchers and docs still point at `RoboMIND1-ur5`, not `robomind1-ur5-joint`. | Export env explicitly or update defaults before launch. |
| Wrong serving entry point | The older RoboLab server assumes Franka 7 joints before gripper and cannot consume 6-joint UR5 observations unchanged. | Use `action_policy_server_robolab_div.py` with `--joint-dof 6`, or port that parameterization into `action_policy_server_robolab.py` before using the non-`_div` module name. |
| Client image contract drift | Current server accepts precomposed `observation/image` or all three RoBoArena wrist/exterior image keys; a single top-camera client message is not enough unless both sides agree on the key/composition path. | Precompose on client, or patch server to build top-view + two black placeholders. |
| Dataset memory pressure | `RoboMINDUR5Dataset` inherits the base row materialization path, which reads all parquet rows into Python dicts (`cosmos_framework/data/vfm/action/datasets/base_dataset.py:76`). This dataset has about 2.97M rows in one parquet file. | If workers OOM or startup is slow, port DROID's compact column-array index pattern (`cosmos_framework/data/vfm/action/datasets/droid_lerobot_dataset.py:122`) to the UR5 adapter. |
| AV1 decode support | Dataset videos are AV1; `decode_video_frames` must be able to decode them in every worker. | Decode one transformed sample in the target training env before full run. |
| Generic offline action inference | `robomind-ur5-single` is registered as a domain id but intentionally absent from `EMBODIMENT_TO_RAW_ACTION_DIM` (`cosmos_framework/data/vfm/action/domain_utils.py:43`). | Use RoboLab divided server or pass explicit raw width in custom tooling; do not assume generic `inference/action.py` will resolve this domain. |
| Iteration count when scaling out | Full 64-rank global batch makes one pass much shorter than on 8 GPUs; unchanged `MAX_ITER=10000` may overtrain. | Recompute max_iter from windows/effective batch and validate by closed-loop checkpoints. |

## hparams.yaml Review Notes

The colleague YAML correctly identifies the single real camera as `overhead` (`tmps/ur5-single-robomind/hparams.yaml:23`),
which supersedes the earlier wrist-camera assumption in this note.

Items I would carry over:

- Absolute joint-position target from future states: this matches the current adapter's 6+1 joint action path.
- Horizon 32 at 15 fps: this matches the current recipe and preflight result.
- Continuous 1D gripper is a reasonable first-run assumption, with polarity still treated as an empirical variable.
- Closed-loop checkpoint selection is the right evaluation criterion; loss alone is not enough.

Items I would not carry over unchanged:

- The YAML's epoch math uses `n_frames / global_batch`. For this chunked policy dataset, the closer sample count is
  stride-1 action windows, currently about `2965157 - 22044 * 32 = 2259749`. Use
  `max_iter ~= target_passes * num_windows / effective_global_batch`.
- The YAML assumes `global_batch: 32`; the current local H200 wrapper uses `NPROC_PER_NODE=8` and
  `MAX_SAMPLES_PER_BATCH=128`, i.e. an effective batch of about 1024. Under that launch shape, 60000 iterations would
  be far beyond the YAML's intended sample exposure.
- With the current wrapper, one window pass is about `2259749 / 1024 ~= 2207` optimizer steps. The existing
  `MAX_ITER=10000` is already about 4.5 passes; the YAML's suggested 15000-40000-step plateau at batch 32 corresponds
  to roughly 469-1250 steps at batch 1024, not 15000-40000 local-wrapper steps.
- `canvas_hw: [360, 640]` with "NO compositing" is a different visual contract from the current Cosmos recipe, which
  uses the DROID-style concat canvas with missing views zero-padded. Switching to true single-image training is a
  larger change than this run's planned small delta.
- `action_norm: per_joint_quantile` is not in the current train/serve contract. Add it only if the server applies the
  exact inverse transform at inference.
- `filter_idle_frames` and `filter_failed_demos` are good ideas only if those labels exist in the LeRobot conversion
  and the adapter applies them deterministically.
