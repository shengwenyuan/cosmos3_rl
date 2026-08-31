from pathlib import Path

import numpy as np

from tools.rh20t.evaluate_cfg4_ur5_joint_policy import (
    classify_action_window,
    evaluate_arrays,
    select_balanced_indices,
    select_indices,
)


def test_select_indices_is_stable_unique_and_bounded() -> None:
    first = select_indices(100, 16, 7)
    assert first == select_indices(100, 16, 7)
    assert len(first) == len(set(first)) == 16
    assert min(first) >= 0 and max(first) < 100


def test_balanced_selector_keeps_low_motion_gripper_and_motion() -> None:
    samples = []
    for index in range(48):
        action = np.zeros((33, 7), dtype=np.float32)
        if index % 3 == 1:
            action[16:, 6] = 1.0
        elif index % 3 == 2:
            action[:, 0] = np.linspace(0.0, 0.2, 33)
        samples.append({"action": action})

    selected = select_balanced_indices(samples, 12, seed=3, candidate_factor=4)
    categories = {classify_action_window(samples[index]["action"]) for index in selected}

    assert len(selected) == 12
    assert categories == {"low_motion", "gripper_event", "motion"}


def test_evaluate_arrays_accepts_target_scale_and_rejects_unsafe_motion() -> None:
    current = np.zeros((4, 7), dtype=np.float32)
    target = np.zeros((4, 32, 7), dtype=np.float32)
    target[:, :, 0] = np.linspace(0.002, 0.064, 32)
    safe = evaluate_arrays(current, target.copy(), target)

    assert safe["passed"]
    assert safe["p99_step_scale_ratio"] == 1.0
    assert safe["gripper_accuracy"] == 1.0

    prediction = target.copy()
    prediction[:, 0, 0] = 1.0
    unsafe = evaluate_arrays(current, prediction, target)
    assert not unsafe["passed"]
    assert "velocity_violation" in unsafe["failures"]
    assert "acceleration_violation" in unsafe["failures"]


def test_gate_report_is_json_serializable(tmp_path: Path) -> None:
    current = np.zeros((1, 7), dtype=np.float32)
    target = np.zeros((1, 32, 7), dtype=np.float32)
    report = evaluate_arrays(current, target, target)
    output = tmp_path / "gate.json"
    import json

    output.write_text(json.dumps(report), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_gripper_tolerance_records_nominal_overshoot_and_rejects_hard_overshoot() -> None:
    current = np.zeros((1, 7), dtype=np.float32)
    target = np.zeros((1, 32, 7), dtype=np.float32)
    slight_overshoot = target.copy()
    slight_overshoot[..., 6] = 1.02

    tolerated = evaluate_arrays(current, slight_overshoot, target)

    assert tolerated["passed"]
    assert tolerated["counts"]["gripper_nominal_range_violations"] == 32
    assert tolerated["counts"]["gripper_range_violations"] == 0
    assert tolerated["gripper_prediction"]["max"] == np.float32(1.02)

    hard_overshoot = slight_overshoot.copy()
    hard_overshoot[..., 6] = 1.06
    rejected = evaluate_arrays(current, hard_overshoot, target)

    assert not rejected["passed"]
    assert rejected["counts"]["gripper_range_violations"] == 32
    assert "gripper_range_violation" in rejected["failures"]
