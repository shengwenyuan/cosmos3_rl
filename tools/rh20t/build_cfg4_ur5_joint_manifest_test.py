# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.rh20t.build_cfg4_ur5_joint_manifest import (
    BuildConfig,
    build_manifest,
    load_task_policy,
    write_outputs,
)
from tools.rh20t.rh20t_io import EXTERIOR_CAMERAS, PRIMARY_SERIAL

pytestmark = pytest.mark.level(0)


def _write_episode(root: Path, name: str, *, rating: int = 9) -> Path:
    episode = root / name
    transformed = episode / "transformed"
    transformed.mkdir(parents=True)
    timestamps = np.arange(0, 4100, 100, dtype=np.int64)
    joint = {int(timestamp): np.linspace(0.0, 0.5, 6) + index * 0.001 for index, timestamp in enumerate(timestamps)}
    gripper = {
        int(timestamp): {"gripper_command": [85.0 - index, 77, 200], "gripper_info": [84, 0, 0]}
        for index, timestamp in enumerate(timestamps)
    }
    np.save(transformed / "joint.npy", {PRIMARY_SERIAL: joint})
    np.save(transformed / "gripper.npy", {PRIMARY_SERIAL: gripper})
    (episode / "metadata.json").write_text(
        json.dumps({"finish_time": int(timestamps[-1]), "rating": rating, "calib_quality": 2})
    )
    for serial in EXTERIOR_CAMERAS:
        camera = episode / f"cam_{serial}"
        camera.mkdir()
        np.save(camera / "timestamps.npy", {"color": timestamps})
        (camera / "color.mp4").write_bytes(b"non-empty-test-placeholder")
    return episode


def test_build_manifest_keeps_fixed_rate_c32_segments_and_holds(tmp_path: Path) -> None:
    source = tmp_path / "RH20T_cfg4"
    source.mkdir()
    _write_episode(source, "task_0034_user_0017_scene_0004_cfg_0004")
    _write_episode(source, "task_0034_user_0017_scene_0005_cfg_0004", rating=1)
    task_file = tmp_path / "task_description.json"
    task_file.write_text(
        json.dumps(
            {
                "task_0034": {
                    "task_description_english": "Stack the squares into a pyramid shape",
                    "task_description_chinese": "将方块堆叠成金字塔形状",
                }
            }
        )
    )

    summary, artifacts, reports = build_manifest(
        source_root=source,
        task_descriptions_path=task_file,
        config=BuildConfig(),
    )

    accepted = artifacts["train"] + artifacts["val"] + artifacts["test"]
    assert summary["counts"]["accepted_source_episodes"] == 1
    assert len(accepted) == 1
    assert accepted[0]["frame_count"] == 61
    assert accepted[0]["window_count_c32"] == 29
    assert accepted[0]["joint_target_source"] == "measured_joint_proxy"
    assert summary["rejection_reasons"] == {"low_or_missing_rating": 1}
    assert reports["gripper_semantics"]["status"] == "pass"

    output = tmp_path / "derived"
    write_outputs(output, summary, artifacts, reports)
    assert (output / "manifests" / "source_inventory.jsonl").is_file()
    assert (output / "reports" / "window_stats_c32.json").is_file()
    stats = json.loads((output / "stats" / "train_quantile_7d.json").read_text())
    assert stats["metadata"]["split"] == "train"
    assert stats["metadata"]["action_layout"][-1] == "gripper_close_fraction"
    assert len(stats["global"]["q01"]) == 7
    checksums = json.loads((output / "checksums.json").read_text())
    assert checksums["sha256"]["manifests/train.jsonl"]
    assert "task_0034" in (output / "task_catalog.csv").read_text()


def test_task_policy_requires_complete_disjoint_decisions(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "policy_id": "test-policy",
                "included_task_ids": ["task_0034"],
                "excluded_tasks": {"task_0011": "liquid_handling"},
            }
        )
    )
    value, included, excluded = load_task_policy(policy, {"task_0011", "task_0034"})
    assert value["policy_id"] == "test-policy"
    assert included == {"task_0034"}
    assert excluded == {"task_0011": "liquid_handling"}

    with pytest.raises(ValueError, match="mismatch"):
        load_task_policy(policy, {"task_0011", "task_0034", "task_0101"})
