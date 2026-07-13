# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.data.generator.action.datasets.berkeley_ur5_eef_dataset import (
    _validate_berkeley_action_policy_manifest,
)
from cosmos_framework.data.generator.action.policy_schema import ActionPolicyManifest, load_action_policy_manifest

pytestmark = pytest.mark.level(0)


def test_berkeley_adapter_rejects_manifest_semantic_drift() -> None:
    manifest = load_action_policy_manifest(
        Path(__file__).parents[5] / "examples/toml/sft_config/action_policy_berkeley_ur5_eef_repro.toml"
    )
    config = SimpleNamespace(view_description=manifest.datasets[0].view_description)
    _validate_berkeley_action_policy_manifest(config, manifest)

    raw = manifest.model_dump(mode="json")
    raw["datasets"][0]["camera_features"] = {
        "primary": "observation.images.image",
        "aux_left": "observation.images.hand_image",
    }
    with pytest.raises(ValueError, match="camera_features|canvas_views"):
        _validate_berkeley_action_policy_manifest(config, ActionPolicyManifest.model_validate(raw))

    raw = manifest.model_dump(mode="json")
    raw["model_action"]["frame"] = "wrong_tcp"
    raw["wire_action"]["frame"] = "wrong_tcp"
    with pytest.raises(ValueError, match="model_frame|wire_frame"):
        _validate_berkeley_action_policy_manifest(config, ActionPolicyManifest.model_validate(raw))
