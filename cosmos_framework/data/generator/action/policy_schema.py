# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Versioned action-policy contract shared by training and serving.

The manifest is intentionally independent from any concrete dataset class or
robot implementation. Training writes it once at the run root; serving
loads it without reflecting over the training dataloader; clients consume only
the derived wire contract advertised during the WebSocket handshake.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Literal

import pydantic
import tomllib
import yaml

GripperSemantics = Literal["close_fraction", "open_fraction"]
EEF_DELTA_LAYOUT = (
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
)
EEF_ABSOLUTE_LAYOUT = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
OBSERVATION_VIEW_SLOTS = ("primary", "aux_left", "aux_right")
_CONFIG_MISSING = object()


class _StrictModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class GripperSpec(_StrictModel):
    index: int = pydantic.Field(ge=0)
    semantics: GripperSemantics


class ActionRepresentation(_StrictModel):
    """One model-side or wire-side action representation."""

    codec: Literal["joint_position", "eef_delta", "eef_absolute"]
    layout: tuple[str, ...]
    representation: Literal["absolute", "delta"]
    gripper: GripperSpec
    frame: str | None = None
    quaternion_order: Literal["xyzw", "wxyz"] | None = None

    @pydantic.model_validator(mode="after")
    def _validate_layout(self) -> "ActionRepresentation":
        if not self.layout:
            raise ValueError("action layout must not be empty")
        if len(set(self.layout)) != len(self.layout):
            raise ValueError("action layout entries must be unique")
        if self.gripper.index >= len(self.layout):
            raise ValueError(f"gripper index {self.gripper.index} is outside action layout width {len(self.layout)}")
        if self.layout[self.gripper.index] != "gripper":
            raise ValueError("the action layout entry at gripper.index must be named 'gripper'")
        if self.codec == "eef_absolute" and self.representation != "absolute":
            raise ValueError("eef_absolute requires representation='absolute'")
        if self.codec == "eef_delta" and self.representation != "delta":
            raise ValueError("eef_delta requires representation='delta'")
        if self.codec == "joint_position" and self.representation != "absolute":
            raise ValueError("joint_position requires representation='absolute'")
        if self.codec == "eef_delta" and self.layout != EEF_DELTA_LAYOUT:
            raise ValueError(f"eef_delta layout must be {EEF_DELTA_LAYOUT!r}")
        if self.codec == "eef_absolute" and self.layout != EEF_ABSOLUTE_LAYOUT:
            raise ValueError(f"eef_absolute layout must be {EEF_ABSOLUTE_LAYOUT!r}")
        if self.codec.startswith("eef_") and self.frame is None:
            raise ValueError("EEF action representations require an explicit frame")
        if self.codec == "eef_absolute" and self.quaternion_order is None:
            raise ValueError("absolute EEF wire actions require quaternion_order")
        if self.codec == "eef_absolute" and self.quaternion_order != "xyzw":
            raise ValueError("action-policy schema v1 supports only quaternion_order='xyzw'")
        return self


class ConditioningSpec(_StrictModel):
    """Temporal meaning of rows supplied to the action model."""

    state_rows: Literal[0, 1]
    history_rows: int = pydantic.Field(ge=0)
    source: Literal["none", "current_state"]
    timing: str

    @pydantic.model_validator(mode="after")
    def _validate_rows(self) -> "ConditioningSpec":
        if self.history_rows < self.state_rows:
            raise ValueError("history_rows must be >= state_rows")
        if (self.state_rows == 0) != (self.source == "none"):
            raise ValueError("source must be 'none' exactly when state_rows is 0")
        if not self.timing.strip():
            raise ValueError("conditioning timing description must not be empty")
        return self


class ObservationSpec(_StrictModel):
    """Machine-checkable canvas geometry shared by all declared sources."""

    layout_id: str
    view_shape_hw: tuple[int, int]
    canvas_shape_hw: tuple[int, int]
    view_roles: tuple[str, ...]
    missing_view_policy: Literal["black", "error"] = "black"
    viewpoint: str = "concat_view"
    description: str

    @pydantic.model_validator(mode="after")
    def _validate_observation(self) -> "ObservationSpec":
        if any(value <= 0 for value in (*self.view_shape_hw, *self.canvas_shape_hw)):
            raise ValueError("view_shape_hw and canvas_shape_hw values must be positive")
        if not self.layout_id.strip() or not self.view_roles:
            raise ValueError("observation layout_id and view_roles must not be empty")
        if self.view_roles != OBSERVATION_VIEW_SLOTS:
            raise ValueError(f"action-policy schema v1 view_roles must be positional slots {OBSERVATION_VIEW_SLOTS!r}")
        view_height, view_width = self.view_shape_hw
        if self.canvas_shape_hw != (view_height + view_height // 2, view_width):
            raise ValueError("schema v1 canvas_shape_hw must be one full view above two half-size views")
        if not self.description.strip():
            raise ValueError("observation description must not be empty")
        return self


class TransformSpec(_StrictModel):
    """ActionTransformPipeline arguments that must match training exactly."""

    resolution: str | None
    max_action_dim: int = pydantic.Field(gt=0)
    action_channel_masking: bool = True
    append_viewpoint_info: bool = True
    append_duration_fps_timestamps: bool = True
    append_resolution_info: bool = True
    append_idle_frames: bool = False
    format_prompt_as_json: bool = False


class NormalizationSpec(_StrictModel):
    """Action normalization supported by manifest schema v1.

    Affine policies need an exact, source-aware stats binding in both training
    and serving. Schema v1 rejects them instead of pretending a YAML copy is
    authoritative while the dataset loads different mutable statistics.
    """

    kind: Literal["none"] = "none"


class DatasetSourceDescription(_StrictModel):
    """Storage provenance plus the source-specific training view prompt."""

    name: str
    root: str
    condition_source: Literal["action_t0", "observation_state_t0", "none"]
    action_features: tuple[str, ...]
    state_features: tuple[str, ...] = ()
    camera_features: dict[str, str]
    action_layout: tuple[str, ...]
    gripper_semantics: GripperSemantics
    source_frame: str | None = None
    source_quaternion_order: Literal["xyzw", "wxyz"] | None = None
    source_gripper_index: int | None = pydantic.Field(default=None, ge=0)
    source_target_offset: int = pydantic.Field(default=0, ge=0)
    description: str
    view_description: str

    @pydantic.model_validator(mode="after")
    def _validate_source(self) -> "DatasetSourceDescription":
        if not self.name.strip() or not self.description.strip() or not self.view_description.strip():
            raise ValueError("dataset source name, description, and view_description must not be empty")
        if not self.action_features or not self.camera_features or not self.action_layout:
            raise ValueError("dataset source action/camera/layout descriptions must not be empty")
        if "primary" not in self.camera_features or not set(self.camera_features) <= set(OBSERVATION_VIEW_SLOTS):
            raise ValueError(
                f"dataset source camera_features must contain 'primary' and use only {OBSERVATION_VIEW_SLOTS!r}"
            )
        if len(set(self.action_layout)) != len(self.action_layout):
            raise ValueError("dataset source action_layout entries must be unique")
        if self.condition_source == "observation_state_t0" and not self.state_features:
            raise ValueError("observation_state_t0 requires state_features")
        if self.condition_source != "observation_state_t0" and self.state_features:
            raise ValueError("state_features are only valid with observation_state_t0")
        if self.source_frame is not None and not self.source_frame.strip():
            raise ValueError("source_frame must be non-empty when provided")
        if self.source_quaternion_order is not None and self.source_frame is None:
            raise ValueError("source_quaternion_order requires source_frame")
        if self.source_gripper_index is None and self.source_target_offset != 0:
            raise ValueError("source_target_offset requires source_gripper_index")
        return self


class ActionPolicyManifest(_StrictModel):
    """Single source of truth for one trained action-policy artifact."""

    schema_version: Literal[1] = 1
    profile_id: str
    robot: str
    domain_name: str
    policy_fps: int = pydantic.Field(gt=0)
    chunk_size: int = pydantic.Field(gt=0)
    model_action: ActionRepresentation
    wire_action: ActionRepresentation
    conditioning: ConditioningSpec
    observation: ObservationSpec
    transform: TransformSpec
    normalization: NormalizationSpec = pydantic.Field(default_factory=NormalizationSpec)
    datasets: tuple[DatasetSourceDescription, ...]

    @pydantic.model_validator(mode="after")
    def _validate_contract(self) -> "ActionPolicyManifest":
        if not self.profile_id.strip() or not self.robot.strip() or not self.domain_name.strip():
            raise ValueError("profile_id, robot, and domain_name must not be empty")
        if not self.datasets:
            raise ValueError("action-policy manifests require at least one dataset source description")
        dataset_names = [source.name for source in self.datasets]
        if len(set(dataset_names)) != len(dataset_names):
            raise ValueError("action-policy dataset source names must be unique")
        if self.wire_action.gripper.semantics != "close_fraction":
            raise ValueError("wire gripper semantics are canonical: close_fraction (0=open, 1=closed)")
        if self.transform.append_idle_frames:
            raise ValueError(
                "action-policy schema v1 requires append_idle_frames=false because future-action idle labels "
                "are unavailable during serving"
            )
        if self.model_action_dim > self.transform.max_action_dim:
            raise ValueError("model action width must not exceed transform.max_action_dim")
        if self.model_action.codec == self.wire_action.codec == "joint_position":
            if self.model_action.layout != self.wire_action.layout:
                raise ValueError("action-policy schema v1 requires identical joint-position model and wire layouts")
            if (
                self.conditioning.state_rows,
                self.conditioning.history_rows,
                self.conditioning.source,
            ) != (1, 1, "current_state"):
                raise ValueError("schema v1 joint-position policies require 1/1 current-state conditioning")
        elif self.model_action.codec == "joint_position":
            raise ValueError("joint-position model actions require a joint-position wire representation")
        if self.model_action.codec == "eef_delta" and self.wire_action.codec != "eef_absolute":
            raise ValueError("EEF delta model actions must declare an absolute EEF wire representation")
        if self.model_action.codec == "eef_delta" and (
            self.conditioning.state_rows,
            self.conditioning.history_rows,
            self.conditioning.source,
        ) != (0, 0, "none"):
            raise ValueError("schema v1 EEF policies require 0/0 no-state conditioning")
        if self.model_action.codec == "eef_delta" and self.model_action.frame != self.wire_action.frame:
            raise ValueError("EEF model and wire frames must match; schema v1 does not perform frame transforms")
        if self.model_action.codec not in {"joint_position", "eef_delta"}:
            raise ValueError(f"unsupported model action codec for schema v1: {self.model_action.codec!r}")
        for source in self.datasets:
            if len(source.action_layout) != self.model_action_dim:
                raise ValueError(
                    f"dataset source {source.name!r} action layout width must match the model action width"
                )
            if (source.condition_source == "none") != (self.conditioning.state_rows == 0):
                raise ValueError(
                    f"dataset source {source.name!r} conditioning does not match the model conditioning rows"
                )
        return self

    @property
    def model_action_dim(self) -> int:
        return len(self.model_action.layout)

    @property
    def wire_action_dim(self) -> int:
        return len(self.wire_action.layout)

    def resolve_dataset_source(self, name: str | None = None) -> DatasetSourceDescription:
        """Resolve the one source contract selected for a serving process."""

        if name is None:
            if len(self.datasets) == 1:
                return self.datasets[0]
            choices = [source.name for source in self.datasets]
            raise ValueError(f"multi-source policies require an explicit dataset source; choose one of {choices!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("dataset source must be a non-empty string")
        for source in self.datasets:
            if source.name == name:
                return source
        choices = [source.name for source in self.datasets]
        raise ValueError(f"unknown dataset source {name!r}; choose one of {choices!r}")

    def client_contract(self, dataset_source: str | None = None) -> dict[str, Any]:
        """Return the public WebSocket contract without private storage paths."""

        selected_source = self.resolve_dataset_source(dataset_source)
        return {
            "protocol_version": self.schema_version,
            "profile_id": self.profile_id,
            "robot": self.robot,
            "action_space": self.wire_action.codec,
            "policy_fps": self.policy_fps,
            "chunk_size": self.chunk_size,
            "wire_action_dim": self.wire_action_dim,
            "action_layout": list(self.wire_action.layout),
            "gripper_indices": [self.wire_action.gripper.index],
            "gripper_semantics": self.wire_action.gripper.semantics,
            "eef_frame": self.wire_action.frame,
            "quaternion_order": self.wire_action.quaternion_order,
            "pose_mode": self.wire_action.representation,
            "conditioning": {
                "state_rows": self.conditioning.state_rows,
                "history_rows": self.conditioning.history_rows,
                "source": self.conditioning.source,
            },
            "observation": {
                "layout_id": self.observation.layout_id,
                "view_shape_hw": list(self.observation.view_shape_hw),
                "canvas_shape_hw": list(self.observation.canvas_shape_hw),
                "view_roles": list(self.observation.view_roles),
                "missing_view_policy": self.observation.missing_view_policy,
                "viewpoint": self.observation.viewpoint,
                "description": self.observation.description,
            },
            # Dataset roots and storage fields stay private to training.  The
            # source name is the stable selector chosen once at server startup;
            # the handshake lets the client validate its matching camera
            # capability. The selected description is the exact prompt text
            # used for that source during training.
            "dataset_sources": [
                {
                    "name": source.name,
                    "description": source.description,
                    "view_description": source.view_description,
                }
                for source in self.datasets
            ],
            "requires_dataset_source": len(self.datasets) > 1,
            "dataset_source": selected_source.name,
            "source_view_description": selected_source.view_description,
            "present_view_roles": [role for role in OBSERVATION_VIEW_SLOTS if role in selected_source.camera_features],
        }


def coerce_action_policy_manifest(value: Any) -> ActionPolicyManifest:
    if isinstance(value, ActionPolicyManifest):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if not isinstance(value, Mapping):
        raise TypeError(f"action policy manifest must be a mapping, got {type(value).__name__}")
    return ActionPolicyManifest.model_validate(dict(value))


def load_action_policy_manifest(path: str | Path) -> ActionPolicyManifest:
    manifest_path = Path(path).expanduser()
    if manifest_path.suffix.lower() == ".toml":
        with manifest_path.open("rb") as file:
            raw = tomllib.load(file)
    else:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "action_policy" in raw:
        raw = raw["action_policy"]
    return coerce_action_policy_manifest(raw)


def save_action_policy_manifest(manifest: ActionPolicyManifest, path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def _config_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _set_config_value(value: Any, key: str, new_value: Any) -> None:
    if isinstance(value, MutableMapping):
        value[key] = new_value
    else:
        setattr(value, key, new_value)


def _find_rank_partitioned_action_config(config: Any) -> Any | None:
    dataloader = _config_value(_config_value(config, "dataloader_train"), "dataloader")
    datasets = _config_value(dataloader, "datasets")
    if not isinstance(datasets, Mapping):
        return None
    candidates = []
    for entry in datasets.values():
        dataset = _config_value(entry, "dataset")
        if _config_value(dataset, "chunk_length") is not None and _config_value(dataset, "max_action_dim") is not None:
            candidates.append(dataset)
    if len(candidates) > 1:
        raise ValueError("An action-policy manifest must describe exactly one canonical action dataset factory")
    return candidates[0] if candidates else None


def validate_training_manifest_alignment(config: Any, manifest: ActionPolicyManifest) -> None:
    """Reject drift between the resolved training factory and artifact YAML."""

    dataset = _find_rank_partitioned_action_config(config)
    if dataset is None:
        return
    model_config = _config_value(_config_value(config, "model"), "config")
    model_max_action_dim = _config_value(model_config, "max_action_dim", _CONFIG_MISSING)
    if model_max_action_dim != manifest.transform.max_action_dim:
        raise ValueError(
            f"action_policy.transform.max_action_dim={manifest.transform.max_action_dim} does not match "
            f"model.config.max_action_dim={model_max_action_dim!r}"
        )
    tokenizer = _config_value(model_config, "tokenizer")
    exact_durations = _config_value(tokenizer, "encode_exact_durations", _CONFIG_MISSING)
    expected_durations = [manifest.chunk_size + 1]
    if exact_durations is _CONFIG_MISSING or list(exact_durations) != expected_durations:
        raise ValueError(
            f"action_policy.chunk_size={manifest.chunk_size} requires model tokenizer "
            f"encode_exact_durations={expected_durations!r}, got {exact_durations!r}"
        )
    factory = _config_value(dataset, "_target_")
    factory_validator = getattr(factory, "action_policy_manifest_validator", None)
    if factory_validator is not None:
        factory_validator(dataset, manifest)
    checks = {
        "fps": manifest.policy_fps,
        "chunk_length": manifest.chunk_size,
        "resolution": manifest.transform.resolution,
        "max_action_dim": manifest.transform.max_action_dim,
        "action_channel_masking": manifest.transform.action_channel_masking,
        "append_viewpoint_info": manifest.transform.append_viewpoint_info,
        "append_duration_fps_timestamps": manifest.transform.append_duration_fps_timestamps,
        "append_resolution_info": manifest.transform.append_resolution_info,
        "append_idle_frames": manifest.transform.append_idle_frames,
        "format_prompt_as_json": manifest.transform.format_prompt_as_json,
    }
    for field, expected in checks.items():
        actual = _config_value(dataset, field, _CONFIG_MISSING)
        if actual is _CONFIG_MISSING:
            raise ValueError(f"resolved action dataset factory does not expose manifest-bound field {field!r}")
        if actual != expected:
            raise ValueError(f"action_policy.{field}={expected!r} does not match resolved dataset value {actual!r}")

    action_normalization = _config_value(dataset, "action_normalization")
    if manifest.normalization.kind == "none" and action_normalization is not None:
        raise ValueError("manifest normalization is none but the resolved dataset enables action normalization")

    mode = _config_value(dataset, "mode", _CONFIG_MISSING)
    if mode is _CONFIG_MISSING:
        raise ValueError("resolved action dataset factory does not expose manifest-bound field 'mode'")
    if mode != "policy":
        raise ValueError(f"action-policy training requires dataset mode='policy', got {mode!r}")
    viewpoint = _config_value(dataset, "viewpoint", _CONFIG_MISSING)
    if viewpoint is _CONFIG_MISSING:
        raise ValueError("resolved action dataset factory does not expose manifest-bound field 'viewpoint'")
    if viewpoint != manifest.observation.viewpoint:
        raise ValueError(
            f"action_policy.observation.viewpoint={manifest.observation.viewpoint!r} "
            f"does not match resolved dataset viewpoint {viewpoint!r}"
        )
    use_state = _config_value(dataset, "use_state")
    if use_state is not None and bool(use_state) != bool(manifest.conditioning.state_rows):
        raise ValueError(
            f"action_policy.conditioning.state_rows={manifest.conditioning.state_rows} "
            f"does not match resolved dataset use_state={use_state!r}"
        )
    action_space = _config_value(dataset, "action_space")
    if action_space is not None:
        codec_aliases = {
            "joint_pos": "joint_position",
            "midtrain": "eef_delta",
            "ee_pose_delta": "eef_delta",
        }
        resolved_codec = codec_aliases.get(action_space, action_space)
        if resolved_codec != manifest.model_action.codec:
            raise ValueError(
                f"action_policy.model_action.codec={manifest.model_action.codec!r} "
                f"does not match resolved dataset action_space={action_space!r}"
            )

    root = _config_value(dataset, "root")
    if root is not None and len(manifest.datasets) == 1 and str(root) != manifest.datasets[0].root:
        raise ValueError(
            f"action_policy.datasets[0].root={manifest.datasets[0].root!r} "
            f"does not match resolved dataset root {str(root)!r}"
        )
    dataset_profile = _config_value(dataset, "dataset_profile", _CONFIG_MISSING)
    if dataset_profile is not _CONFIG_MISSING and len(manifest.datasets) == 1:
        if dataset_profile != manifest.datasets[0].name:
            raise ValueError(
                f"action_policy.datasets[0].name={manifest.datasets[0].name!r} "
                f"does not match resolved dataset_profile={dataset_profile!r}"
            )
    canvas_views = _config_value(dataset, "canvas_views", _CONFIG_MISSING)
    if canvas_views is not _CONFIG_MISSING and len(manifest.datasets) == 1:
        cameras = manifest.datasets[0].camera_features
        expected_views = tuple(cameras[role] for role in ("primary", "aux_left", "aux_right") if role in cameras)
        actual_views = tuple(canvas_views) if canvas_views is not None else None
        if actual_views != expected_views:
            raise ValueError(
                f"action_policy.datasets[0].camera_features imply canvas_views={expected_views!r}, "
                f"but the resolved dataset has {actual_views!r}"
            )
    gripper_invert = _config_value(dataset, "gripper_invert")
    if gripper_invert is not None and len(manifest.datasets) == 1:
        source_semantics = manifest.datasets[0].gripper_semantics
        resolved_model_semantics = (
            ("open_fraction" if source_semantics == "close_fraction" else "close_fraction")
            if gripper_invert
            else source_semantics
        )
        if resolved_model_semantics != manifest.model_action.gripper.semantics:
            raise ValueError(
                "action_policy model/source gripper semantics do not match the resolved "
                f"gripper_invert={gripper_invert!r}"
            )
    decode_size_hw = _config_value(dataset, "decode_size_hw")
    if decode_size_hw is not None and tuple(decode_size_hw) != manifest.observation.view_shape_hw:
        raise ValueError(
            f"action_policy.observation.view_shape_hw={manifest.observation.view_shape_hw!r} "
            f"does not match resolved dataset decode_size_hw={tuple(decode_size_hw)!r}"
        )


def _factory_parameters(dataset: Any) -> set[str]:
    factory = _config_value(dataset, "_target_")
    if not callable(factory):
        raise ValueError("manifest-bound action dataset _target_ must be a callable factory")
    return set(inspect.signature(factory).parameters)


def _bind_required_factory_values(dataset: Any, parameters: set[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if key not in parameters:
            raise ValueError(f"manifest-bound action dataset factory does not accept required field {key!r}")
        _set_config_value(dataset, key, value)


def bind_manifest_to_training_config(config: Any, manifest: ActionPolicyManifest) -> None:
    """Bind the validated manifest into the concrete training factory.

    Policy/transform values are always injected, including values omitted from
    the experiment's LazyCall.  A factory may then either consume the generic
    multi-source ``sources`` protocol or a single dedicated source.  This makes
    the run-root YAML the value owner rather than an audit copy of Python
    defaults.
    """

    dataset = _find_rank_partitioned_action_config(config)
    if dataset is None:
        return
    tokenizer = _config_value(_config_value(_config_value(config, "model"), "config"), "tokenizer")
    if tokenizer is None:
        raise ValueError("manifest-bound action training requires model.config.tokenizer")
    _set_config_value(tokenizer, "encode_exact_durations", [manifest.chunk_size + 1])
    parameters = _factory_parameters(dataset)
    _bind_required_factory_values(
        dataset,
        parameters,
        {
            "fps": float(manifest.policy_fps),
            "chunk_length": manifest.chunk_size,
            "mode": "policy",
            "viewpoint": manifest.observation.viewpoint,
            "action_normalization": None,
            "resolution": manifest.transform.resolution,
            "max_action_dim": manifest.transform.max_action_dim,
            "action_channel_masking": manifest.transform.action_channel_masking,
            "append_viewpoint_info": manifest.transform.append_viewpoint_info,
            "append_duration_fps_timestamps": manifest.transform.append_duration_fps_timestamps,
            "append_resolution_info": manifest.transform.append_resolution_info,
            "append_idle_frames": manifest.transform.append_idle_frames,
            "format_prompt_as_json": manifest.transform.format_prompt_as_json,
        },
    )

    if "sources" not in parameters:
        if len(manifest.datasets) != 1:
            raise ValueError("dedicated action dataset factories accept exactly one manifest dataset source")
        source = manifest.datasets[0]
        _bind_required_factory_values(
            dataset,
            parameters,
            {
                "root": source.root,
                "view_description": source.view_description,
            },
        )
        if "dataset_profile" in parameters:
            _set_config_value(dataset, "dataset_profile", source.name)
        if "canvas_views" in parameters:
            roles = ("primary", "aux_left", "aux_right")
            _set_config_value(
                dataset,
                "canvas_views",
                tuple(source.camera_features[role] for role in roles if role in source.camera_features),
            )
        if "decode_size_hw" in parameters:
            _set_config_value(dataset, "decode_size_hw", manifest.observation.view_shape_hw)
        return

    factory = _config_value(dataset, "_target_")
    source_binder = getattr(factory, "action_policy_manifest_binder", None)
    if source_binder is None:
        raise ValueError(
            "multi-source action dataset factories must expose an action_policy_manifest_binder(manifest) hook"
        )
    source_configs = source_binder(manifest)
    if not isinstance(source_configs, (list, tuple)) or not source_configs:
        raise ValueError("action-policy dataset source binder must return a non-empty source sequence")
    _set_config_value(dataset, "sources", list(source_configs))


def find_action_policy_manifest(checkpoint_path: str | Path) -> Path | None:
    """Find the canonical run sidecar from a local model/checkpoint path.

    Discovery accepts only an artifact's own sidecar or the owner of a standard
    ``run/checkpoints/...`` tree. Other layouts must pass ``--policy-config``;
    an arbitrary parent directory must never become artifact semantics.
    """

    if "://" in str(checkpoint_path):
        return None
    path = Path(checkpoint_path).expanduser().absolute()
    start = path if path.is_dir() else path.parent
    checkpoint_owner: Path | None = None
    for directory in (start, *start.parents):
        if directory.name == "checkpoints":
            checkpoint_owner = directory.parent
            break
    candidates = [checkpoint_owner] if checkpoint_owner is not None else [start]
    for directory in candidates:
        sidecar = directory / "action_policy.yaml"
        if sidecar.is_file():
            return sidecar
    return None


def _find_run_action_policy_manifest(run_dir: Path) -> Path | None:
    """Find only manifests owned by ``run_dir`` (never an ancestor run)."""

    candidate = run_dir / "action_policy.yaml"
    return candidate if candidate.is_file() else None


def persist_run_action_policy(
    config: Any,
    *,
    allow_legacy_adoption: bool = False,
) -> ActionPolicyManifest | None:
    """Validate and persist ``config.action_policy`` before training writes.

    If a run already has checkpoints, changing (or newly inventing) its
    action contract is rejected before the trainer can overwrite config.yaml.
    """

    value = getattr(config, "action_policy", None)
    if value is None:
        if getattr(config, "requires_action_policy_manifest", False):
            raise ValueError(
                "This action-policy experiment requires an explicit [action_policy] section in its training TOML"
            )
        return None
    manifest = coerce_action_policy_manifest(value)
    validate_training_manifest_alignment(config, manifest)
    config.action_policy = manifest.model_dump(mode="json")

    run_dir = Path(config.job.path_local).expanduser()
    checkpoints_dir = run_dir / "checkpoints"
    latest_checkpoint = checkpoints_dir / "latest_checkpoint.txt"
    has_checkpoints = latest_checkpoint.exists() or any(checkpoints_dir.glob("iter_*"))
    # Resume ownership is deliberately narrower than serving discovery: a new
    # sibling/child run must never inherit a parent run's contract merely
    # because ``find_action_policy_manifest`` walks checkpoint ancestors.
    existing_path = _find_run_action_policy_manifest(run_dir) if run_dir.exists() else None
    if existing_path is not None:
        existing = load_action_policy_manifest(existing_path)
        if existing != manifest and has_checkpoints:
            raise ValueError(
                "Refusing to resume an action-policy run with a different manifest: "
                f"existing={existing_path}. Choose a new [job].name/output directory."
            )
    elif has_checkpoints and not allow_legacy_adoption:
        raise ValueError(
            "Existing checkpoints have no action_policy manifest. Refusing to attach new semantics during resume; "
            "pass --adopt-legacy-action-policy-manifest after auditing the explicit TOML, or choose a new "
            "[job].name/output directory."
        )

    # This is the only canonical owner for the run. Checkpoint paths discover
    # it by walking to the run; config.yaml may retain an audit snapshot but is
    # never used as serving/resume semantics.
    save_action_policy_manifest(manifest, run_dir / "action_policy.yaml")
    return manifest


__all__ = [
    "ActionPolicyManifest",
    "ActionRepresentation",
    "ConditioningSpec",
    "DatasetSourceDescription",
    "GripperSemantics",
    "GripperSpec",
    "NormalizationSpec",
    "OBSERVATION_VIEW_SLOTS",
    "ObservationSpec",
    "TransformSpec",
    "coerce_action_policy_manifest",
    "bind_manifest_to_training_config",
    "find_action_policy_manifest",
    "load_action_policy_manifest",
    "persist_run_action_policy",
    "save_action_policy_manifest",
    "validate_training_manifest_alignment",
]
