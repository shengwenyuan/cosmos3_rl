# RoboMIND1 UR5 single-arm post-training plan (superseded)

This historical design was replaced on 2026-07-13 by the source-neutral,
manifest-driven action-policy path. It is intentionally kept as a short
migration marker rather than preserving obsolete server commands and polarity
switches.

The active implementation is:

- Training experiment:
  `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_ur5_single_joint_nano.py`
- Multi-source LeRobot adapter:
  `cosmos_framework/data/generator/action/datasets/ur5_single_lerobot_dataset.py`
- Explicit run contract:
  `examples/toml/sft_config/action_policy_robomind_ur5_single_repro.toml`
- Universal server:
  `cosmos_framework/scripts/action_policy_server_robolab.py`
- Current operator documentation:
  `docs/action_policy_robomind_ur5_posttrain.md`

Current invariants:

- The model action is 7D absolute joint position: six UR5 arm joints plus one
  gripper channel.
- New UR5 policies use `close_fraction` (`0=open`, `1=closed`) on both model and
  wire sides. Source-specific polarity conversion occurs once in the dataset
  adapter.
- Dataset roots, feature names, camera-role mappings, action order, observation
  canvas, timing, and transform settings are declared in `[action_policy]`.
- Training writes one canonical `action_policy.yaml` at the run root. Serving
  discovers it from checkpoint descendants or accepts an explicit legacy TOML.
- The client selects only robot/camera capabilities; action codec, dimensions,
  FPS, chunking, conditioning, layout, and gripper semantics come from the
  server handshake.

Do not restore robot-specific server flags or dataset-name branching. Add a new
source description or adapter capability when onboarding another dataset or
robot.
