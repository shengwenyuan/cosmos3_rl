# Berkeley UR5 EEF Retraining TODO

This note captures the remaining known fixes before the next full Berkeley UR5
EEF training run. It intentionally leaves these items out of scope for the next
round:

- Wrist camera registration is considered usable for now, even though strict
  Berkeley hand-camera alignment remains a future improvement.
- The dataset does not provide enough evidence to distinguish `tool0` from TCP,
  so the current RoboLab EEF frame should remain unchanged unless new evidence
  appears.
- Gripper registration and control are handled on the client side; this document
  focuses on Cosmos training and server representation alignment.

## 1. Align The Image Canvas With DROID

Current Berkeley SFT data uses `top=external`, `bottom-left=wrist`, and
`bottom-right=zero`. DROID-style concat-view policies use `top=wrist` and
bottom external views. The next Berkeley full run should switch Berkeley UR5 EEF
to the DROID-style visual contract:

- Training adapter:
  - Update `cosmos_framework/data/vfm/action/datasets/berkeley_ur5_eef_dataset.py`
    so `_load_concat_video()` produces `top=wrist`, `bottom-left=external`, and
    `bottom-right=zero`.
  - Update the class docstring and `additional_view_description` text in the
    same file.
  - Prefer configuring `canvas_views=("observation.images.hand_image",
    "observation.images.image")` in
    `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_berkeley_ur5_eef_nano.py`
    once the adapter order is updated and validated.
- Server:
  - Update `_CONCAT_VIEW_DESCRIPTION` in
    `cosmos_framework/scripts/action_policy_server_robolab_div.py` to describe
    `top=wrist`, `bottom-left=external`, `bottom-right=zero`.
- Client:
  - Update `Cosmos3UR5Client._compose_canvas()` in the RoboLab client so the
    actual request image matches the training canvas.
- Validation:
  - Save one training sample canvas and one live RoboLab request canvas.
  - Verify both have the same view order, aspect, padding, and prompt text before
    launching a full run.

## 4. Train On Standard SE(3) Delta Actions

The current Berkeley adapter trains on the native Berkeley command stream:
`[native_xyz, rot6d(native_rpy), gripper]`. The next full run should instead
derive the pose action from `observation.state` absolute EEF poses:

- Read `observation.state[:3]` as position and `observation.state[3:7]` as
  `quat_xyzw`.
- Build an absolute pose trajectory for the `chunk_length + 1` observation rows
  using `build_abs_pose_from_components(..., "quat_xyzw")`.
- Build the supervised pose target with
  `pose_abs_to_rel(..., rotation_format="rot6d", pose_convention="backward_framewise")`.
- Concatenate one gripper channel to keep the model target width at 10D:
  `[se3_delta_translation(3), se3_delta_rot6d(6), gripper(1)]`.
- Decide and document the gripper source explicitly:
  - If using Berkeley `action[:, 6]`, preserve the current action-stream target
    convention and deployment mapping.
  - If using `observation.state[:, 7]`, validate lag and inversion against the
    client-side gripper command convention before training.

Acceptance checks:

- The new action shape remains `(chunk_length, 10)`.
- Reconstructing absolute poses with `pose_rel_to_abs()` from the first EEF
  state matches the next `observation.state` poses within a small tolerance.
- The old temporary native-command fit is no longer needed for deployment.

## 5. Restore Standard Server EEF Decoding

The divided RoboLab server currently contains a temporary Berkeley native-command
decoder for the existing checkpoint. A checkpoint trained on standard SE(3)
deltas should use the normal inverse conversion:

- Remove or bypass `_temporary_berkeley_native_command_to_abs_eef_pose()` in
  `cosmos_framework/scripts/action_policy_server_robolab_div.py` for the new
  checkpoint path.
- In the `eef_pose` response branch:
  - Build `initial_pose` from the latest `observation/eef_pos` and
    `observation/eef_quat`.
  - Decode model output with
    `pose_rel_to_abs(action_np[:, :9], rotation_format="rot6d", pose_convention="backward_framewise", initial_pose=initial_pose)`.
  - Return the client 8D absolute EEF contract:
    `[position(3), quat_xyzw(4), gripper(1)]`.
- Keep `--action-space eef_pose --action-dim 10` as the raw model-output
  contract; the server response remains 8D after decoding.
- Re-check the gripper inversion line in the server after the new training
  target is chosen. It should be a documented convention, not a hidden fixup.

## 6. Keep State And History Semantics Consistent

The current Berkeley dataset emits `chunk_length` action rows for
`chunk_length + 1` video frames and does not prepend a state/action history row.
Deployment should keep:

```bash
--no-use-state --history-length 0
```

unless the training dataset is intentionally changed to include state or history
conditioning.

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
