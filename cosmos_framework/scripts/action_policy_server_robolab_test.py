# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the manifest-driven RoboLab action-policy server."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import ActionProcessingRecord
from cosmos_framework.data.generator.action.policy_schema import ActionPolicyManifest, save_action_policy_manifest

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    for module_name in (
        "cosmos_framework.scripts.action_policy_server_utils",
        "cosmos_framework.scripts.action_policy_server_robolab",
    ):
        sys.modules.pop(module_name, None)
    from cosmos_framework.scripts import action_policy_server_robolab as robolab_server  # noqa: E402

pytestmark = pytest.mark.level(0)


def _manifest(
    *,
    robot: str = "ur5",
    joints: int = 6,
    model_gripper: str = "close_fraction",
    history_rows: int = 1,
) -> ActionPolicyManifest:
    layout = [*(f"joint_{index}" for index in range(joints)), "gripper"]
    return ActionPolicyManifest.model_validate(
        {
            "schema_version": 1,
            "profile_id": f"{robot}_joint_test",
            "robot": robot,
            "domain_name": "ur5-single-joint" if robot == "ur5" else "droid_lerobot",
            "policy_fps": 15,
            "chunk_size": 4,
            "model_action": {
                "codec": "joint_position",
                "layout": layout,
                "representation": "absolute",
                "gripper": {"index": joints, "semantics": model_gripper},
            },
            "wire_action": {
                "codec": "joint_position",
                "layout": layout,
                "representation": "absolute",
                "gripper": {"index": joints, "semantics": "close_fraction"},
            },
            "conditioning": {
                "state_rows": 1,
                "history_rows": history_rows,
                "source": "current_state",
                "timing": "row 0 is current state; remaining rows are future targets",
            },
            "observation": {
                "layout_id": "test_three_view",
                "view_shape_hw": [3, 5],
                "canvas_shape_hw": [4, 5],
                "view_roles": ["primary", "aux_left", "aux_right"],
                "missing_view_policy": "black",
                "viewpoint": "concat_view",
                "description": "one primary slot above two auxiliary slots",
            },
            "transform": {
                "resolution": None,
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
                    "name": "test",
                    "root": "/tmp/test",
                    "condition_source": "action_t0",
                    "action_features": ["action"],
                    "state_features": [],
                    "camera_features": {"primary": "camera"},
                    "action_layout": layout,
                    "gripper_semantics": model_gripper,
                    "description": "Explicit source fixture.",
                    "view_description": "custom wrist camera and two shoulder views",
                }
            ],
        }
    )


def _service_config(manifest: ActionPolicyManifest) -> robolab_server.RobolabPolicyConfig:
    return robolab_server.RobolabPolicyConfig(
        checkpoint_path="/unused/model",
        manifest=manifest,
        dataset_source=manifest.resolve_dataset_source(),
        decode_video=False,
        seed=0,
        deterministic_seed=True,
        guidance=3.0,
        num_steps=4,
        shift=5.0,
    )


def _eef_manifest() -> ActionPolicyManifest:
    raw = _manifest(history_rows=1).model_dump(mode="json")
    raw.update(profile_id="ur5_eef_test", chunk_size=4)
    raw["model_action"] = {
        "codec": "eef_delta",
        "layout": [
            "delta_x",
            "delta_y",
            "delta_z",
            "rot6d_0",
            "rot6d_1",
            "rot6d_2",
            "rot6d_3",
            "rot6d_4",
            "rot6d_5",
            "gripper",
        ],
        "representation": "delta",
        "frame": "berkeley_tcp",
        "gripper": {"index": 9, "semantics": "close_fraction"},
    }
    raw["wire_action"] = {
        "codec": "eef_absolute",
        "layout": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        "representation": "absolute",
        "frame": "berkeley_tcp",
        "quaternion_order": "xyzw",
        "gripper": {"index": 7, "semantics": "close_fraction"},
    }
    raw["conditioning"] = {
        "state_rows": 0,
        "history_rows": 0,
        "source": "none",
        "timing": "four future SE(3) deltas without a state row",
    }
    raw["datasets"][0].update(
        condition_source="none",
        action_features=["observation.state", "action"],
        state_features=[],
        action_layout=raw["model_action"]["layout"],
    )
    return ActionPolicyManifest.model_validate(raw)


def test_resolve_public_hf_policy_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloaded_path = tmp_path / "downloaded"
    downloaded_path.mkdir()
    calls: list[tuple[str, str]] = []

    def fake_download(checkpoint: Any) -> str:
        calls.append((checkpoint.repository, checkpoint.revision))
        return str(downloaded_path)

    monkeypatch.setattr(robolab_server.CheckpointDirHf, "download", fake_download)
    resolved = robolab_server._resolve_checkpoint_path("Cosmos3-Nano-Policy-DROID", hf_revision="test")

    assert resolved == str(downloaded_path)
    assert calls == [("nvidia/Cosmos3-Nano-Policy-DROID", "test")]


def test_resolve_checkpoint_keeps_existing_local_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model"
    checkpoint_path.mkdir()
    assert robolab_server._resolve_checkpoint_path(str(checkpoint_path), hf_revision="main") == str(checkpoint_path)


def test_validate_checkpoint_accepts_diffusers_safetensors_index(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")
    robolab_server._validate_checkpoint(str(tmp_path), allow_dcp_checkpoint=False)


def test_policy_manifest_is_discovered_or_uses_explicit_droid_release(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_000000001" / "model"
    checkpoint.mkdir(parents=True)
    expected = _manifest()
    sidecar = tmp_path / "run" / "action_policy.yaml"
    save_action_policy_manifest(expected, sidecar)

    found = robolab_server._resolve_policy_manifest(
        str(checkpoint), requested_checkpoint=str(checkpoint), policy_config=None
    )
    assert found == expected
    assert (
        robolab_server._resolve_policy_manifest(
            str(checkpoint), requested_checkpoint=str(checkpoint), policy_config=sidecar
        )
        == expected
    )
    conflicting = _manifest(robot="future_arm")
    conflicting_path = tmp_path / "conflicting.yaml"
    save_action_policy_manifest(conflicting, conflicting_path)
    with pytest.raises(ValueError, match="conflicts with"):
        robolab_server._resolve_policy_manifest(
            str(checkpoint), requested_checkpoint=str(checkpoint), policy_config=conflicting_path
        )

    droid = robolab_server._resolve_policy_manifest(
        str(tmp_path / "downloaded"),
        requested_checkpoint="nvidia/Cosmos3-Nano-Policy-DROID",
        policy_config=None,
    )
    assert droid.robot == "droid"
    assert droid.model_action.gripper.semantics == "open_fraction"
    assert droid.wire_action.gripper.semantics == "close_fraction"


def test_missing_manifest_fails_instead_of_guessing_dataset_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No action-policy manifest"):
        robolab_server._resolve_policy_manifest(
            str(tmp_path / "model"), requested_checkpoint=str(tmp_path / "model"), policy_config=None
        )


def test_load_openpi_websocket_policy_server_from_lightweight_package(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWebsocketPolicyServer:
        pass

    fake_package = type(sys)("openpi_server")
    fake_package.__path__ = []
    fake_module = type(sys)("openpi_server.websocket_policy_server")
    fake_module.WebsocketPolicyServer = FakeWebsocketPolicyServer
    monkeypatch.setitem(sys.modules, "openpi_server", fake_package)
    monkeypatch.setitem(sys.modules, "openpi_server.websocket_policy_server", fake_module)

    assert robolab_server._load_openpi_websocket_policy_server() is FakeWebsocketPolicyServer


def test_server_args_only_default_runtime_not_policy_semantics() -> None:
    args = robolab_server.RobolabServerArgs()
    assert args.checkpoint_path == "nvidia/Cosmos3-Nano-Policy-DROID"
    assert args.hf_revision == "main"
    assert args.policy_config is None
    assert args.dataset_source is None
    assert args.seed == 0
    assert args.guidance == 3.0
    assert args.num_steps == 4
    assert args.shift == 5.0
    assert not hasattr(args, "robot")
    assert not hasattr(args, "gripper_invert")


def test_policy_contract_is_manifest_driven_and_allows_arbitrary_robot_name() -> None:
    manifest = _manifest(robot="future_arm")
    manifest = manifest.model_copy(update={"domain_name": "ur5-single-joint"})
    contract = robolab_server._build_policy_contract(_service_config(manifest))

    assert contract["robot"] == "future_arm"
    assert contract["action_layout"][-1] == "gripper"
    assert contract["gripper_semantics"] == "close_fraction"
    assert contract["observation"]["layout_id"] == "test_three_view"
    assert contract["conditioning"]["history_rows"] == 1
    assert contract["dataset_source"] == "test"
    assert contract["source_view_description"] == "custom wrist camera and two shoulder views"
    assert contract["present_view_roles"] == ["primary"]


def test_build_transform_uses_manifest_not_training_dataloader() -> None:
    service = object.__new__(robolab_server.RobolabPolicyService)
    service.model = SimpleNamespace(config=SimpleNamespace(max_action_dim=64, vlm_config=None))
    transform = service._build_transform(_manifest())
    assert transform.max_action_dim == 64
    assert transform.prompt_json_formatter is not None
    assert transform.text_tokenizer is None


def test_server_requires_client_composed_observation_canvas() -> None:
    with pytest.raises(ValueError, match="composition belongs to the client"):
        robolab_server._extract_observation_image(
            {
                "observation/wrist_image_left": np.zeros((4, 5, 3), dtype=np.uint8),
                "observation/exterior_image_1_left": np.zeros((4, 5, 3), dtype=np.uint8),
                "observation/exterior_image_2_left": np.zeros((4, 5, 3), dtype=np.uint8),
            }
        )


@pytest.mark.parametrize(
    ("model_semantics", "expected_gripper"),
    [("close_fraction", 0.3), ("open_fraction", 0.7)],
)
def test_joint_observation_converts_wire_close_fraction_once(model_semantics: str, expected_gripper: float) -> None:
    manifest = _manifest(model_gripper=model_semantics)
    service = object.__new__(robolab_server.RobolabPolicyService)
    service.cfg = _service_config(manifest)
    service._transform = lambda sample, resolution, action_normalizer=None: sample

    image = np.zeros((4, 5, 3), dtype=np.uint8)
    joint_position = np.arange(12, dtype=np.float32).reshape(2, 6)
    gripper_position = np.array([[0.2], [0.3]], dtype=np.float32)
    sample = service._build_sample(
        {
            "prompt": "pick the cube",
            "observation/image": image,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
        }
    )

    assert sample["video"].shape == (3, 5, 4, 5)
    assert sample["action"].shape == (5, 7)
    np.testing.assert_allclose(sample["action"][0, :6], joint_position[-1])
    assert sample["action"][0, 6].item() == pytest.approx(expected_gripper)
    assert "history_action" not in sample
    assert sample["additional_view_description"] == manifest.datasets[0].view_description


def test_gripper_codec_is_explicit() -> None:
    values = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        robolab_server._convert_gripper_semantics(values, "open_fraction", "close_fraction"),
        1.0 - values,
    )


def test_eef_manifest_returns_full_absolute_wire_horizon() -> None:
    manifest = _eef_manifest()
    service = object.__new__(robolab_server.RobolabPolicyService)
    service.cfg = _service_config(manifest)
    service._lock = threading.Lock()

    def transform(sample, resolution, action_normalizer=None):
        sample["action_processing_record"] = ActionProcessingRecord(raw_action_dim=10, action_normalizer=None)
        return sample

    # OmniMoTModel returns actions already unpadded and denormalized.
    model_action = torch.zeros((4, 10), dtype=torch.float32)
    model_action[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    model_action[:, 9] = 0.75
    service._transform = transform
    service.model = SimpleNamespace(generate_samples_from_batch=lambda *args, **kwargs: {"action": [model_action]})
    result = service.infer(
        {
            "prompt": "move",
            "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
            "observation/eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "observation/eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "observation/gripper_position": np.array([0.25], dtype=np.float32),
        }
    )

    assert result["action"].shape == (4, 8)
    np.testing.assert_allclose(result["action"][:, :3], [[0.1, 0.2, 0.3]] * 4, atol=1e-6)
    np.testing.assert_allclose(result["action"][:, 7], 0.75)


def test_build_data_batch_wraps_multi_item_keys_like_internal_server() -> None:
    sample = {
        "video": torch.zeros((3, 2, 4, 5), dtype=torch.uint8),
        "action": torch.zeros((1, 8), dtype=torch.float32),
        "domain_id": torch.tensor(1, dtype=torch.long),
        "conditioning_fps": torch.tensor(15, dtype=torch.long),
        "ai_caption": "move",
    }
    batch = robolab_server._build_data_batch_from_sample(sample)
    assert batch["video"][0][0] is sample["video"]
    assert batch["action"][0][0] is sample["action"]
    assert batch["domain_id"][0].shape == (1,)
    assert batch["conditioning_fps"][0].shape == (1,)
    assert batch["ai_caption"] == ["move"]


def test_serve_advertises_only_manifest_wire_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    config = _service_config(_manifest())

    class FakeServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def serve_forever(self):
            captured["served"] = True

    monkeypatch.setattr(robolab_server, "RobolabPolicyService", lambda args: SimpleNamespace(cfg=config))
    monkeypatch.setattr(robolab_server, "_load_openpi_websocket_policy_server", lambda: FakeServer)
    monkeypatch.setattr(robolab_server, "get_local_ip", lambda: "127.0.0.1")
    robolab_server.serve(robolab_server.RobolabServerArgs())

    contract = captured["metadata"]["policy_contract"]
    assert contract["gripper_semantics"] == "close_fraction"
    assert contract["wire_action_dim"] == 7
    assert contract["conditioning"]["source"] == "current_state"
    assert "datasets" not in contract
    assert captured["served"] is True
