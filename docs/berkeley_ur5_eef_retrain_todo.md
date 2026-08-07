# Berkeley UR5 EEF Retraining TODO

This note captures the Berkeley UR5 EEF fixes and validation items before the
next full training run. Items 1, 4, and 5 have code-level fixes in place; keep the
validation checks below before treating a new checkpoint as production-ready.

The following items remain intentionally out of scope for the next round:

- Wrist camera registration is considered usable for now, even though strict
  Berkeley hand-camera alignment remains a future improvement.
- The dataset does not provide enough evidence to distinguish `tool0` from TCP,
  so the current RoboLab EEF frame should remain unchanged unless new evidence
  appears.
- Gripper registration and control are handled on the client side; this document
  focuses on Cosmos training and server representation alignment.

## 1. Align The Image Canvas With DROID

Berkeley SFT now follows the DROID-style concat-view contract:
`top=wrist`, `bottom-left=external`, and `bottom-right=zero`. Keep the explicit
validation checks below before launching the next Berkeley full run:

- Training adapter:
  - `cosmos_framework/data/generator/action/datasets/berkeley_ur5_eef_dataset.py`
    resolves `canvas_views` as wrist first, external second, then zero-fills the
    bottom-right quadrant.
  - The class docstring and `additional_view_description` describe the same
    view order.
  - `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_berkeley_ur5_eef_nano.py`
    configures `canvas_views=("observation.images.hand_image",
    "observation.images.image")`.
- Server:
  - `action_policy.observation` in the training TOML and persisted sidecar
    describes `top=wrist`, `bottom-left=external`, `bottom-right=zero`.
- Client:
  - `Cosmos3UR5Client._compose_canvas()` in the RoboLab client must use the same
    request image layout: top wrist, bottom-left external, bottom-right zero.
- Validation:
  - Save one training sample canvas and one live RoboLab request canvas.
  - Verify both have the same view order, aspect, padding, and prompt text before
    launching a full run.

## 4. Train On Standard SE(3) Delta Actions

The Berkeley adapter now derives the pose action from `observation.state`
absolute EEF poses instead of training on the native Berkeley command stream
`[native_xyz, rot6d(native_rpy), gripper]`:

- `observation.state[:3]` is read as position and `observation.state[3:7]` as
  `quat_xyzw`.
- The adapter builds an absolute pose trajectory for the `chunk_length + 1`
  observation rows using `build_abs_pose_from_components(..., "quat_xyzw")`.
- The supervised pose target is built with
  `pose_abs_to_rel(..., rotation_format="rot6d", pose_convention="backward_framewise")`.
- One native Berkeley action-stream gripper channel, `action[:, 6]`, is
  concatenated to keep the model target width at 10D:
  `[se3_delta_translation(3), se3_delta_rot6d(6), gripper(1)]`.
- If a future run switches gripper supervision to `observation.state[:, 7]`,
  validate lag and inversion against the client-side command convention before
  training.

Acceptance checks:

- The new action shape remains `(chunk_length, 10)`.
- Reconstructing absolute poses with `pose_rel_to_abs()` from the first EEF
  state matches the next `observation.state` poses within a small tolerance.
- The old temporary native-command fit is no longer needed for deployment.

## 5. Restore Standard Server EEF Decoding

The manifest-driven RoboLab server now uses the standard inverse conversion for
checkpoints trained on SE(3) deltas. Older native-command Berkeley checkpoints
should not use this path without their temporary decoder:

- `_temporary_berkeley_native_command_to_abs_eef_pose()` is removed from the new
  checkpoint path in `cosmos_framework/scripts/action_policy_server_robolab.py`.
- In the `eef_delta -> eef_absolute` codec:
  - Build `initial_pose` from the latest `observation/eef_pos` and
    `observation/eef_quat`.
  - Decode model output with
    `pose_rel_to_abs(action_np[:, :9], rotation_format="rot6d", pose_convention="backward_framewise", initial_pose=initial_pose)`.
  - Return the client 8D absolute EEF contract:
    `[position(3), quat_xyzw(4), gripper(1)]`.
- Keep the manifest's 10D `model_action` and 8D `wire_action` layouts aligned;
  the response remains 8D after decoding.
- Model and wire gripper semantics are explicit manifest fields; wire semantics
  are always canonical `close_fraction`.

## 6. Keep State And History Semantics Consistent

The current Berkeley dataset emits `chunk_length` action rows for
`chunk_length + 1` video frames and does not prepend a state/action history row.
The manifest therefore declares `state_rows=0` and `history_rows=0`. There is
no server CLI override for these training semantics.

If history is added later:

- Add `history_action` in the training sample path, not only in the server.
- Ensure `ActionTransformPipeline` receives the same number of history rows in
  training and serving.
- Make `condition_frame_indexes_action` and server output trimming match the
  trained sequence plan.
- Add a smoke test that checks returned action horizon equals the client
  open-loop horizon after trimming.

## 8. Recompute Action Statistics And Normalization Policy

The previous Berkeley run used `action_normalization=None`, so the model learned
raw native-command values. Switching to standard SE(3) deltas changes the action
distribution.

Before training:

- Generate a short stats report for the new 10D target:
  - mean/std/min/max/q01/q99 per channel
  - gripper open/close distribution
  - reconstructed-pose error from the SE(3) round trip
- Decide whether to keep `action_normalization=None` or enable an affine
  normalizer.
- If enabling normalization:
  - Add or regenerate dataset stats under the dataset stats path.
  - Confirm the training transform records `raw_action_dim=10`.
  - Confirm inference postprocessing denormalizes back to raw SE(3) deltas before
    server-side `pose_rel_to_abs()`.
- If keeping raw actions:
  - Confirm delta translation and rotation magnitudes are numerically stable for
    the action head.
  - Compare loss scale against the previous Berkeley run before launching the
    full 8x H20 job.

## Minimal Pre-Run Checklist

1. Render and inspect one Berkeley training canvas after the top/bottom-left
   swap.
2. Render and inspect one RoboLab live request canvas from the client.
3. Run an offline SE(3) round-trip validation over several Berkeley episodes.
4. Run server-client banana smoke with the new checkpoint and standard decoder.
5. Only then use task success rate as a policy-quality signal.
