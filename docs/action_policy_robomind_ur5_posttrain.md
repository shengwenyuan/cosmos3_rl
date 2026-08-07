# Cosmos3-Nano RoboMIND1-UR Joint-Space Post-Training

This document covers **Case B** from `tmps/UR5_WORKING_PIPELINE.md`: RoboMIND 1.0 UR joint-space post-training for a UR5 target embodiment.

Berkeley AUTOLab UR5 is now a separate **Case A** EEF-space path with its own dataset adapter and recipe. Do not train Berkeley through this RoboMIND joint-space recipe.

## Case B Summary

| item | value |
| --- | --- |
| dataset | RoboMIND 1.0 UR subset converted to LeRobot |
| expected path | `/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint` |
| action | **7D** `[joint(6), gripper(1)]` absolute UR joint targets |
| domain | `ur5-single-joint` |
| experiment | `action_policy_robomind_ur5_single_nano` |
| recipe TOML | `examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml` |
| launcher | `examples/launch_sft_action_policy_robomind_ur5_single.sh` |

The 7D joint head is fresh. Do not reuse DROID's 8D Franka joint head.

## Dataset Path

RoboMIND1-UR LeRobot data is expected under the fast-disk LeRobot area:

```bash
export DATASET_PATH=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint
export UR5_SINGLE_ROOT=$DATASET_PATH
```

The launcher default is already set to this path, but exporting the variables keeps runs explicit.

## Required Dataset Schema

The source-neutral UR5 reader accepts either one packed 7D action field or
explicit split joint/gripper fields. The training TOML must describe the exact
storage mapping; filenames and polarity are not inferred from a dataset name.

This RoboMIND recipe declares the following split fields:

```text
action.arm_left_joint               # shape [6]
action.gripper_left                 # shape [1] or scalar
observation.state.arm_left_joint     # shape [6]
observation.state.gripper_left       # shape [1] or scalar
observation.images.camera_top        # at least one camera is required
```

Other sources may declare different feature names or a packed `action` field in
their own `[[action_policy.datasets]]` entry. Missing camera roles follow the
manifest's explicit `missing_view_policy`.

## Canvas Policy

The joint-space recipe uses a fixed three-view canvas:

```text
top row:        real overview/top camera
bottom-left:    second selected camera, if present; otherwise zeros
bottom-right:   third selected camera, if present; otherwise zeros
```

This keeps the visual layout stable even when RoboMIND1-UR only provides one camera.

## Training Setup

```bash
cd /root/code/cosmos-framework
source /root/.bashrc
cosmos3-activate
export LD_LIBRARY_PATH=

export DATASET_PATH=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/robomind1-ur5-joint
export UR5_SINGLE_ROOT=$DATASET_PATH
export BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir>
export WAN_VAE_PATH=<Wan2.2_VAE.pth>
export IMAGINAIRE_OUTPUT_ROOT=/mlp_vepfs/share/swy/cosmos3-framework/outputs/train
```

Run after the dataset is migrated:

```bash
bash examples/launch_sft_action_policy_robomind_ur5_single.sh
```

Smoke override after data/checkpoints exist:

```bash
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=8"
bash examples/launch_sft_action_policy_robomind_ur5_single.sh
```

## Config Notes

| setting | value |
| --- | --- |
| base LR | `2e-4` for the full multi-node batch shape |
| action-head multipliers | `5x` for `action2llm`, `llm2action`, `action_modality_embed` |
| fresh params | action encoder, decoding MLP, action embedding tokens, action positional embeddings |
| action normalization | `None` for raw absolute joint targets |
| W&B | TOML sets `wandb_mode = "online"`; export `WANDB_API_KEY` if logging is desired |

Scale LR down for smaller effective batch sizes.

## Deployment

RoboLab joint-space deployment is direct:

```text
model 7D output -> UR5 joint targets + gripper scalar -> RoboLab client
```

No IK is needed. Joint order and canonical `close_fraction` gripper semantics
are checked against the client capability during the server handshake.
RoboLab evaluation must run `policies/cosmos3/run_ur5.py` with
`--camera-preset robomind_single`, which maps the overhead camera and generates
the two black auxiliary views declared by this recipe.

## Separate Berkeley EEF Path

Berkeley AUTOLab UR5 uses a different recipe:

```text
examples/toml/sft_config/action_policy_berkeley_ur5_eef_repro.toml
examples/launch_sft_action_policy_berkeley_ur5_eef.sh
```

That path converts Berkeley `action[7]` delta-RPY into a 10D EEF delta action. Local Berkeley EEF validation passes under `tmps/berkeley_eef_validation/summary.md`; RoboLab evaluation still requires the EEF-to-joint IK deployment bridge.

## Code Pointers

- Single-arm joint dataset: `cosmos_framework/data/generator/action/datasets/ur5_single_lerobot_dataset.py`
- Legacy dual-arm joint dataset: `cosmos_framework/data/generator/action/datasets/robomind_ur5_dataset.py`
- Berkeley EEF dataset: `cosmos_framework/data/generator/action/datasets/berkeley_ur5_eef_dataset.py`
- Canvas helper: `cosmos_framework/data/generator/action/datasets/canvas_utils.py`
- Joint experiment: `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robomind_ur5_single_nano.py`
- Berkeley EEF experiment: `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_berkeley_ur5_eef_nano.py`
