# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset import (
    get_action_ur5_single_sft_dataset,
)
from cosmos_framework.data.generator.action.policy_schema import (
    ActionPolicyManifest,
    bind_manifest_to_training_config,
    find_action_policy_manifest,
    load_action_policy_manifest,
    persist_run_action_policy,
    save_action_policy_manifest,
    validate_training_manifest_alignment,
)

pytestmark = pytest.mark.level(0)


def _fake_action_factory(
    *,
    sources=None,
    fps=15.0,
    chunk_length=32,
    mode="policy",
    viewpoint="concat_view",
    action_normalization=None,
    resolution="480",
    max_action_dim=64,
    action_channel_masking=True,
    append_viewpoint_info=True,
    append_duration_fps_timestamps=True,
    append_resolution_info=True,
    append_idle_frames=False,
    format_prompt_as_json=True,
):
    raise AssertionError("configuration-only test factory must not be instantiated")


def _raw_manifest() -> dict:
    layout = ["joint_0", "joint_1", "gripper"]
    return {
        "schema_version": 1,
        "profile_id": "test_joint",
        "robot": "test_arm",
        "domain_name": "ur5-single-joint",
        "policy_fps": 15,
        "chunk_size": 32,
        "model_action": {
            "codec": "joint_position",
            "layout": layout,
            "representation": "absolute",
            "gripper": {"index": 2, "semantics": "close_fraction"},
        },
        "wire_action": {
            "codec": "joint_position",
            "layout": layout,
            "representation": "absolute",
            "gripper": {"index": 2, "semantics": "close_fraction"},
        },
        "conditioning": {
            "state_rows": 1,
            "history_rows": 1,
            "source": "current_state",
            "timing": "current state then 32 future targets",
        },
        "observation": {
            "layout_id": "three_view",
            "view_shape_hw": [360, 640],
            "canvas_shape_hw": [540, 640],
            "view_roles": ["primary", "aux_left", "aux_right"],
            "missing_view_policy": "black",
            "viewpoint": "concat_view",
            "description": "Explicit three-view canvas.",
        },
        "transform": {
            "resolution": "480",
            "max_action_dim": 64,
            "action_channel_masking": True,
            "append_viewpoint_info": True,
            "append_duration_fps_timestamps": True,
            "append_resolution_info": True,
            "append_idle_frames": False,
            "format_prompt_as_json": True,
        },
        "normalization": {"kind": "none"},
        "datasets": [
            {
                "name": "source",
                "root": "/data/source",
                "condition_source": "observation_state_t0",
                "action_features": ["action.joint", "action.gripper"],
                "state_features": ["observation.joint", "observation.gripper"],
                "camera_features": {"primary": "observation.images.wrist"},
                "action_layout": layout,
                "gripper_semantics": "close_fraction",
                "description": "Split storage mapped to the canonical policy.",
                "view_description": "Explicit three-view canvas.",
            }
        ],
    }


def _config_with_dataset(**values):
    values.setdefault("_target_", _fake_action_factory)
    values.setdefault("fps", 15.0)
    values.setdefault("chunk_length", 32)
    values.setdefault("mode", "policy")
    values.setdefault("viewpoint", "concat_view")
    values.setdefault("action_normalization", None)
    values.setdefault("resolution", "480")
    values.setdefault("max_action_dim", 64)
    values.setdefault("action_channel_masking", True)
    values.setdefault("append_viewpoint_info", True)
    values.setdefault("append_duration_fps_timestamps", True)
    values.setdefault("append_resolution_info", True)
    values.setdefault("append_idle_frames", False)
    values.setdefault("format_prompt_as_json", True)
    dataset = SimpleNamespace(**values)
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                max_action_dim=64,
                tokenizer=SimpleNamespace(encode_exact_durations=[33]),
            )
        ),
        dataloader_train=SimpleNamespace(
            dataloader=SimpleNamespace(datasets={"policy": SimpleNamespace(dataset=dataset)})
        ),
    )


def test_manifest_yaml_roundtrip_and_checkpoint_discovery(tmp_path: Path) -> None:
    manifest = ActionPolicyManifest.model_validate(_raw_manifest())
    run = tmp_path / "run"
    path = save_action_policy_manifest(manifest, run / "action_policy.yaml")
    checkpoint = run / "checkpoints" / "iter_000000150" / "model"
    checkpoint.mkdir(parents=True)
    stale = _raw_manifest()
    stale["profile_id"] = "stale_checkpoint_copy"
    save_action_policy_manifest(ActionPolicyManifest.model_validate(stale), checkpoint / "action_policy.yaml")

    assert load_action_policy_manifest(path) == manifest
    assert find_action_policy_manifest(checkpoint) == path
    assert manifest.client_contract()["gripper_semantics"] == "close_fraction"
    assert manifest.client_contract()["conditioning"] == {
        "state_rows": 1,
        "history_rows": 1,
        "source": "current_state",
    }


def test_wire_gripper_must_be_canonical_close_fraction() -> None:
    raw = _raw_manifest()
    raw["wire_action"]["gripper"]["semantics"] = "open_fraction"
    with pytest.raises(ValueError, match="canonical"):
        ActionPolicyManifest.model_validate(raw)


def test_dataset_source_selection_is_explicit_for_multi_source_policies() -> None:
    raw = _raw_manifest()
    second = dict(raw["datasets"][0])
    second.update(name="second", root="/data/second", view_description="Second camera prompt.")
    raw["datasets"].append(second)
    manifest = ActionPolicyManifest.model_validate(raw)

    with pytest.raises(ValueError, match="explicit dataset source"):
        manifest.resolve_dataset_source()
    assert manifest.resolve_dataset_source("second").view_description == "Second camera prompt."
    assert manifest.client_contract("second")["dataset_source"] == "second"
    with pytest.raises(ValueError, match="unknown dataset source"):
        manifest.resolve_dataset_source("missing")

    raw["datasets"][1]["name"] = "source"
    with pytest.raises(ValueError, match="names must be unique"):
        ActionPolicyManifest.model_validate(raw)


def test_manifest_validates_layout_and_rejects_unbound_normalization() -> None:
    raw = _raw_manifest()
    raw["model_action"]["gripper"]["index"] = 1
    with pytest.raises(ValueError, match="named 'gripper'"):
        ActionPolicyManifest.model_validate(raw)

    raw = _raw_manifest()
    raw["normalization"] = {"kind": "affine", "offset": [0.0] * 3, "scale": [1.0] * 3}
    with pytest.raises(ValueError, match="Input should be 'none'"):
        ActionPolicyManifest.model_validate(raw)

    raw = _raw_manifest()
    raw["wire_action"]["layout"] = ["joint_1", "joint_0", "gripper"]
    with pytest.raises(ValueError, match="identical joint-position"):
        ActionPolicyManifest.model_validate(raw)

    raw = _raw_manifest()
    raw["conditioning"]["history_rows"] = 2
    with pytest.raises(ValueError, match="1/1 current-state"):
        ActionPolicyManifest.model_validate(raw)

    raw = _raw_manifest()
    raw["transform"]["append_idle_frames"] = True
    with pytest.raises(ValueError, match="unavailable during serving"):
        ActionPolicyManifest.model_validate(raw)


def test_dedicated_dataset_plumbing_is_bound_to_manifest() -> None:
    manifest = ActionPolicyManifest.model_validate(_raw_manifest())

    with pytest.raises(ValueError, match="dataset_profile"):
        validate_training_manifest_alignment(_config_with_dataset(dataset_profile=None), manifest)

    raw = _raw_manifest()
    raw["datasets"][0]["camera_features"]["aux_left"] = "observation.images.left"
    manifest = ActionPolicyManifest.model_validate(raw)
    with pytest.raises(ValueError, match="canvas_views"):
        validate_training_manifest_alignment(
            _config_with_dataset(
                canvas_views=("observation.images.left", "observation.images.wrist"),
            ),
            manifest,
        )


def test_multi_source_binding_preserves_each_training_view_description() -> None:
    repo = Path(__file__).parents[4]
    raw = load_action_policy_manifest(
        repo / "examples/toml/sft_config/action_policy_ur5_single_joint_overfit.toml"
    ).model_dump(mode="json")
    second = dict(raw["datasets"][0])
    second.update(name="second_source", root="/data/second", view_description="Second camera prompt.")
    raw["datasets"].append(second)
    manifest = ActionPolicyManifest.model_validate(raw)
    config = _config_with_dataset(_target_=get_action_ur5_single_sft_dataset, sources=[])

    bind_manifest_to_training_config(config, manifest)
    sources = config.dataloader_train.dataloader.datasets["policy"].dataset.sources

    assert [source["view_description"] for source in sources] == [
        manifest.datasets[0].view_description,
        "Second camera prompt.",
    ]


def test_persist_rejects_manifest_drift_on_resume(tmp_path: Path) -> None:
    config = SimpleNamespace(
        action_policy=_raw_manifest(),
        job=SimpleNamespace(path_local=str(tmp_path / "run")),
    )
    persisted = persist_run_action_policy(config)
    assert persisted is not None
    assert (tmp_path / "run" / "action_policy.yaml").is_file()
    assert not (tmp_path / "run" / "checkpoints" / "action_policy.yaml").exists()
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "latest_checkpoint.txt").write_text("iter_000000001", encoding="utf-8")

    changed = _raw_manifest()
    changed["chunk_size"] = 16
    config.action_policy = changed
    with pytest.raises(ValueError, match="different manifest"):
        persist_run_action_policy(config)


def test_manifest_can_change_before_first_checkpoint(tmp_path: Path) -> None:
    config = SimpleNamespace(
        action_policy=_raw_manifest(),
        job=SimpleNamespace(path_local=str(tmp_path / "run")),
    )
    persist_run_action_policy(config)
    changed = _raw_manifest()
    changed["profile_id"] = "after_dryrun"
    config.action_policy = changed

    assert persist_run_action_policy(config).profile_id == "after_dryrun"


def test_legacy_checkpoint_requires_explicit_manifest_adoption(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    checkpoint = run / "checkpoints" / "iter_000000450"
    checkpoint.mkdir(parents=True)
    config = SimpleNamespace(
        action_policy=_raw_manifest(),
        job=SimpleNamespace(path_local=str(run)),
    )

    with pytest.raises(ValueError, match="adopt-legacy-action-policy-manifest"):
        persist_run_action_policy(config)
    persisted = persist_run_action_policy(config, allow_legacy_adoption=True)
    assert persisted is not None
    assert load_action_policy_manifest(run / "action_policy.yaml") == persisted


def test_required_action_experiment_rejects_missing_manifest(tmp_path: Path) -> None:
    config = SimpleNamespace(
        action_policy=None,
        requires_action_policy_manifest=True,
        job=SimpleNamespace(path_local=str(tmp_path / "run")),
    )
    with pytest.raises(ValueError, match="requires an explicit"):
        persist_run_action_policy(config)


def test_new_run_does_not_inherit_parent_manifest(tmp_path: Path) -> None:
    parent = tmp_path / "runs"
    parent.mkdir()
    unrelated = ActionPolicyManifest.model_validate(_raw_manifest())
    save_action_policy_manifest(unrelated, parent / "action_policy.yaml")

    changed = _raw_manifest()
    changed["profile_id"] = "child_run"
    config = SimpleNamespace(
        action_policy=changed,
        job=SimpleNamespace(path_local=str(parent / "child")),
    )
    persisted = persist_run_action_policy(config)

    assert persisted is not None
    assert persisted.profile_id == "child_run"
    assert load_action_policy_manifest(parent / "child" / "action_policy.yaml") == persisted


def test_serving_discovery_does_not_inherit_arbitrary_parent_manifest(tmp_path: Path) -> None:
    parent = tmp_path / "outputs"
    parent.mkdir()
    save_action_policy_manifest(ActionPolicyManifest.model_validate(_raw_manifest()), parent / "action_policy.yaml")
    checkpoint = parent / "new_run" / "weights"
    checkpoint.mkdir(parents=True)

    assert find_action_policy_manifest(checkpoint) is None


def test_config_yaml_is_only_an_explicit_manifest_source(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config_path = run / "config.yaml"
    config_path.write_text(yaml.safe_dump({"action_policy": _raw_manifest()}), encoding="utf-8")
    checkpoint = run / "checkpoints" / "iter_1"
    checkpoint.mkdir(parents=True)
    assert find_action_policy_manifest(checkpoint) is None
    assert load_action_policy_manifest(config_path).profile_id == "test_joint"


def test_droid_repro_records_source_model_and_wire_gripper_boundaries() -> None:
    repo_root = Path(__file__).parents[4]
    manifest = load_action_policy_manifest(repo_root / "examples/toml/sft_config/action_policy_droid_repro.toml")

    assert manifest.datasets[0].gripper_semantics == "close_fraction"
    assert manifest.model_action.gripper.semantics == "open_fraction"
    assert manifest.wire_action.gripper.semantics == "close_fraction"
