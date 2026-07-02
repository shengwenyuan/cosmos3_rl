# UR5(e) Policy Post-Training — Working Status (transient; deletable)

> Scratch/status note. The **definitive** guide is
> [`docs/action_policy_robomind_ur5_posttrain.md`](../docs/action_policy_robomind_ur5_posttrain.md).
> Delete this once the pipeline is validated on the run host.

## Design (approved): two independent cases, one per RoboMIND download

| case | source | arms | action | domain id | experiment | env |
| --- | --- | --- | --- | --- | --- | --- |
| **single** (RoboLab-bound) | RoboMIND 1.2 `h5_ur_1rgb` | 1× 6-DoF | 7-D `[joint6,grip]` | `robomind-ur5-single` (30) | `action_policy_robomind_ur5_single_nano` | `UR5_SINGLE_ROOT` |
| **dual** (secondary) | RoboMIND 2.0 | 2× 6-DoF | 14-D | `robomind-ur5-dual` (31) | `action_policy_robomind_ur5_dual_nano` | `UR5_DUAL_ROOT` |

Both resume from mid-trained Cosmos3-Nano, fresh action heads + 5× LR (paper §4.2.5). Domain ids at
the top of the 32-slot range (NVIDIA ≤20) leave 21–29 as an upstream buffer. Only
`EMBODIMENT_TO_DOMAIN_ID` is registered — the joint_pos policy width comes from the server's
`action_space`, so `EMBODIMENT_TO_RAW_ACTION_DIM` (ee_pose/FD width) is intentionally left unset,
mirroring DROID's joint_pos.

## Files (done)
- Converter `tools/convert_robomind_hdf5_to_lerobot.py` — auto-detects `ur_1rgb` vs `mind2_dual`,
  one format per run; BGR→RGB for `ur_1rgb`; task-from-path (v1) / `/metadata` (v2); fps from
  timestamps (v2) or `--fps` (v1); `--action-source puppet|master`.
- Dataset `robomind_ur5_dataset.py` — `which_arm ∈ {left,dual}`; auto-canvas from cameras present
  (1-view v1 / 3-view v2); `use_state` prepend; episode-shuffle blocks; gripper raw (flag).
- Recipes: `_robomind_ur5_common.py` builder + `action_policy_robomind_ur5_{single,dual}_nano.py`
  (registered in `configs/base/config.py`).
- `examples/toml/sft_config/action_policy_robomind_ur5_{single,dual}_repro.toml` +
  `examples/launch_sft_action_policy_robomind_ur5_{single,dual}.sh`.
- Registry: `domain_utils.py`, `datasets/__init__.py`.

## Verified statically
py_compile clean; converter↔dataset feature names + domain ids consistent; `mind2` read path
(JPEG decode, `(33,14)` dual / `(33,7)` slice, 540×640 canvas) validated on `tmps/trajectory.hdf5`.

## `TODO(verify)` on the run host / with a real `h5_ur_1rgb` file
1. **lerobot writer API** (`create`/`add_frame`/`save_episode`) for the installed version.
2. **ur_1rgb read**: BGR swap looks right, path-based task parse, and **fps** (no in-file timestamps).
3. **Gripper polarity** — kept raw; set `gripper_invert=True` to match the DROID/`1-g` serving convention.
4. `--dryrun` config load per TOML; 1-node 10-iter smoke; export; serve via the user's UR5e RoboLab.
