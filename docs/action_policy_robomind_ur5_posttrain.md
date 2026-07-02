<!-- UR5e post-training — local addition, not part of upstream Cosmos3. -->

# Cosmos3-Nano RoboMIND UR5(e) Action-Policy Post-Training

Post-train [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) into a UR5(e)
robot-manipulation policy on [RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)
data, the way NVIDIA post-trained [Cosmos3-Nano-Policy-DROID](./action_policy_droid_posttrain.md):
resume from the **mid-trained** Cosmos3-Nano, initialize the action encoder / decode-MLP / embed
tokens **fresh**, apply a **5× LR** on the action params, and predict **absolute joint-position
action chunks** from proprioceptive state + video. Reuses the whole DROID policy stack.

## Two cases (one per RoboMIND download)

| case | source | arms | action | domain | experiment |
| --- | --- | --- | --- | --- | --- |
| **single** | RoboMIND **1.2** `h5_ur_1rgb` | 1× UR5e 6-DoF | **7-D** `[joint(6), gripper(1)]` | `robomind-ur5-single` (id 30) | `action_policy_robomind_ur5_single_nano` |
| **dual** | RoboMIND **2.0** | 2× UR5e 6-DoF | **14-D** `[L_joint(6), L_grip, R_joint(6), R_grip]` | `robomind-ur5-dual` (id 31) | `action_policy_robomind_ur5_dual_nano` |

The **single** case is the RoboLab-bound one: RoboLab / RoboArena evaluate a **single-arm** robot, and
UR5e is 6-DoF → 7-D. (The DROID benchmark robot is a 7-joint Franka; a UR5e policy is a *different*
robot — serving it is handled by the user's own UR5e RoboLab setup.) The **dual** case is a second
embodiment (not directly RoboLab-runnable). Both resume from the mid-trained Cosmos3-Nano with fresh
action heads (paper §4.2.5).

## Prerequisites
- [Setup](../README.md#setup) — training extras (`uv sync --all-extras --group=cu130-train`).
- Converter deps on the run host: `h5py`, `pillow`, `lerobot`.
- Domain ids 30/31 sit at the top of the 32-slot embodiment range (NVIDIA uses ≤20), leaving 21–29
  as a buffer against future upstream domains.

## Step 1 — Convert RoboMIND HDF5 → LeRobot v3 (per case, into its own path)

```shell
# SINGLE (RoboMIND 1.2 h5_ur_1rgb). No in-file fps → pass --fps for your data's rate.
python tools/convert_robomind_hdf5_to_lerobot.py \
  --src /path/to/h5_ur_1rgb --out /path/to/robomind_ur5_single_lerobot/success \
  --repo-id local/robomind_ur5_single --fps 15

# DUAL (RoboMIND 2.0)
python tools/convert_robomind_hdf5_to_lerobot.py \
  --src /path/to/robomind2_ur5 --out /path/to/robomind_ur5_dual_lerobot/success \
  --repo-id local/robomind_ur5_dual
```

The converter **auto-detects** the format per file and **asserts one format per run** (convert each
download separately). Each result must contain `meta/info.json`. Key format specifics:

- **`ur_1rgb`** (RoboMIND 1.2, schema `tmps/mind1_all_robot_h5_info_v1.2.md`): reads
  `puppet/joint_position (T,7)` = 6 UR joints + gripper (single arm → written into the `left` slot);
  single camera `observations/rgb_images/camera_top`, stored **BGR** → swapped to RGB; **task parsed
  from the directory path** (`<task>/success_episodes/…`); **fps must be supplied** (no timestamps).
- **`mind2_dual`** (RoboMIND 2.0, verified on a real file): both `arm_{L,R}_position_align` + grippers,
  cameras `camera_{top,front,wrist_left,wrist_right}` (RGB JPEG), task from `/metadata`, fps from
  timestamps.
- `--action-source {puppet,master}` (default `puppet` = executed); `--bgr {auto,true,false}`;
  `--store-hw 360 640`. The dataset auto-builds the canvas from the cameras present (1 view for
  `ur_1rgb`; 3-view DROID-style front+wrists for RoboMIND 2.0).

> The `mind2_dual` HDF5 read + JPEG decode were validated on a real file. The LeRobot **writer** calls
> and the `ur_1rgb` read (BGR swap, path-based task, fps) carry `TODO(verify)` until exercised on a
> host with `lerobot` and a real `h5_ur_1rgb` file.

## Step 2 — Convert the base checkpoint & launch

```shell
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Nano -o $BASE_CHECKPOINT_PATH

# SINGLE (bridged: DATASET_PATH -> UR5_SINGLE_ROOT)
export DATASET_PATH=/path/to/robomind_ur5_single_lerobot/success
export BASE_CHECKPOINT_PATH=/path/to/base_checkpoint
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export NPROC_PER_NODE=8
bash examples/launch_sft_action_policy_robomind_ur5_single.sh

# DUAL: bash examples/launch_sft_action_policy_robomind_ur5_dual.sh  (DATASET_PATH -> UR5_DUAL_ROOT)
```

Single-node smoke:
```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 \
                             dataloader_train.max_samples_per_batch=8"
bash examples/launch_sft_action_policy_robomind_ur5_single.sh
```
Multi-node HSDP: set `model.parallelism.data_parallel_replicate_degree = <num_nodes>` (shard 8).

## Recipe (both cases)

| knob | value |
| --- | --- |
| init | `Cosmos3-Nano`; action encoder / decode-MLP / embed tokens **fresh** (skipped on load) |
| action | `joint_pos`, raw / un-normalized; `use_state=true` → `(chunk+1, dim)` |
| resolution / canvas | `480`; `concat_view`, auto from cameras present (1 view / 3-view 540×640) |
| chunk length | `32` (`encode_exact_durations=[33]`) |
| lr | `2e-4` (full-batch), **5×** on the action heads; FusedAdam; linear decay |
| shuffle | episode-shuffle stream (grad-norm stability) |

## Checkpoints & Serving
- Saved every `save_iter`; resumable (same `job.name`); export via
  `cosmos_framework.scripts.export_model`.
- **Serving is handled by the user's UR5e RoboLab setup.** Keep in mind the shipped
  `action_policy_server_robolab.py` is hardcoded to 7 Franka joints + a DROID canvas and flips the
  gripper (`1-g`) on output; a 6-joint UR5e needs the user's own server/embodiment. Match the trained
  7-D output + camera canvas to that setup, and confirm gripper polarity there (the dataset keeps the
  gripper raw; set `gripper_invert=True` to match the DROID/`1-g` convention if needed).

## Files (all local additions)
- `tools/convert_robomind_hdf5_to_lerobot.py` — dual-format HDF5 → LeRobot v3 converter.
- `cosmos_framework/data/vfm/action/datasets/robomind_ur5_dataset.py` — `RoboMINDUR5Dataset` + factory.
- `.../action/posttrain_config/_robomind_ur5_common.py` + `action_policy_robomind_ur5_{single,dual}_nano.py`.
- `examples/toml/sft_config/action_policy_robomind_ur5_{single,dual}_repro.toml` +
  `examples/launch_sft_action_policy_robomind_ur5_{single,dual}.sh`.
- Registry edits: `domain_utils.py` — **`EMBODIMENT_TO_DOMAIN_ID`** only (`robomind-ur5-single` id 30,
  `robomind-ur5-dual` id 31; used by training + serving). Intentionally **not** added to
  `EMBODIMENT_TO_RAW_ACTION_DIM` (that dict is the ee_pose/FD-inference width; the joint_pos policy
  takes its width from the server's `action_space`, same as DROID's joint_pos). Plus
  `configs/base/config.py` (registers both experiments) and `datasets/__init__.py` (exports the class).
