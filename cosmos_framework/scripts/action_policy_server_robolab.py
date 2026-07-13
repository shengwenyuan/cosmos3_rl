# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Manifest-driven WebSocket action-policy server for RoboLab/openpi clients.

The server uses OpenPI's WebsocketPolicyServer and speaks its msgpack+NumPy protocol:

- on connection, it advertises the explicit robot/action policy contract;
- each client message is an observation dict;
- each response is a dict with ``action`` and, when enabled, ``video``.

All action, timing, gripper, and observation semantics come from the versioned
``action_policy.yaml`` written by training. Robot names are labels, never a
switch that hard-codes dimensions.

Example:

  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
    --port 8000
"""

# Initialize the script runtime before any cosmos-framework imports, mirroring
# the LIBERO server.  The shared helpers below (single-rank distributed init,
# local IP discovery, frozen-config EMA disable) live in
# ``action_policy_server_utils`` so this module doesn't have to import from
# the sibling LIBERO server just to share runtime utilities.
from cosmos_framework.inference.common.init import init_script

init_script()

import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
import tyro

from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.policy_schema import (
    ActionPolicyManifest,
    DatasetSourceDescription,
    find_action_policy_manifest,
    load_action_policy_manifest,
)
from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import tyro_cli
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf

_DEFAULT_DROID_POLICY_CHECKPOINT = "nvidia/Cosmos3-Nano-Policy-DROID"
_DEFAULT_ROBOLAB_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "robolab"
_DEFAULT_HF_REVISION = "main"
_ROBOLAB_POLICY_HF_REPOSITORIES = {
    "Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
    "nvidia/Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
}

_BUILTIN_DROID_MANIFEST = Path(__file__).with_name("action_policy_manifests") / "droid_release.yaml"


def _resolve_checkpoint_path(checkpoint_path: str, *, hf_revision: str) -> str:
    if Path(checkpoint_path).expanduser().exists():
        return checkpoint_path

    repository = _ROBOLAB_POLICY_HF_REPOSITORIES.get(checkpoint_path)
    if repository is None:
        return checkpoint_path

    log.info(
        f"[robolab-policy-server] downloading consolidated checkpoint from Hugging Face: "
        f"repository={repository!r} revision={hf_revision!r}"
    )
    return CheckpointDirHf(repository=repository, revision=hf_revision).download()


def _resolve_policy_manifest(
    checkpoint_path: str,
    *,
    requested_checkpoint: str,
    policy_config: Path | None,
) -> ActionPolicyManifest:
    discovered = find_action_policy_manifest(checkpoint_path)
    discovered_manifest = load_action_policy_manifest(discovered) if discovered is not None else None
    explicit_manifest = load_action_policy_manifest(policy_config) if policy_config is not None else None
    if discovered_manifest is not None and explicit_manifest is not None and discovered_manifest != explicit_manifest:
        raise ValueError(
            f"Explicit policy config {policy_config} conflicts with the checkpoint owner's canonical {discovered}"
        )
    if discovered_manifest is not None:
        return discovered_manifest
    if explicit_manifest is not None:
        return explicit_manifest
    if requested_checkpoint in _ROBOLAB_POLICY_HF_REPOSITORIES:
        # The released DROID artifact predates sidecars; keep its compatibility
        # contract as one explicit, versioned manifest instead of scattered
        # defaults and gripper booleans.
        return load_action_policy_manifest(_BUILTIN_DROID_MANIFEST)
    raise ValueError(
        "No action-policy manifest found for this checkpoint. Pass --policy-config PATH or place the canonical "
        "action_policy.yaml at the run/export root; server-side dataset guessing has been removed."
    )


def _validate_checkpoint(checkpoint_path: str, *, allow_dcp_checkpoint: bool) -> None:
    if checkpoint_path in OmniSetupOverrides.CHECKPOINTS:
        return
    if "://" in checkpoint_path:
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated local safetensors checkpoint directory, not a DCP path. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )

    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    if any(checkpoint_dir.glob("*.distcp")):
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated safetensors checkpoint, but found a DCP checkpoint. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )
    has_config = (checkpoint_dir / "config.json").exists()
    has_consolidated_safetensors = any(checkpoint_dir.glob("*.safetensors"))
    has_diffusers_safetensors_index = (checkpoint_dir / "model.safetensors.index.json").exists()
    if (
        not checkpoint_dir.is_dir()
        or not has_config
        or not (has_consolidated_safetensors or has_diffusers_safetensors_index)
    ):
        raise ValueError(f"Invalid safetensors checkpoint directory: {checkpoint_dir}")


def _ensure_rgb_uint8_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key!r} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _ensure_2d_float_array(value: Any, key: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{key!r} must have shape [T,D] or [D], got {array.shape}")
    if width is not None and array.shape[-1] != width:
        raise ValueError(f"{key!r} must have width {width}, got {array.shape[-1]}")
    return np.ascontiguousarray(array)


def _ensure_gripper_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != 1:
        raise ValueError(f"'observation/gripper_position' must have shape [T,1], [T], or scalar, got {array.shape}")
    return np.ascontiguousarray(array)


def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)  # [1,3,H2,W2]
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # [H2,W2,3]


def _extract_observation_image(obs: dict[str, Any]) -> np.ndarray:
    try:
        value = obs["observation/image"]
    except KeyError as error:
        raise ValueError(
            "Observation must contain the contract-level composed canvas at 'observation/image'; "
            "camera-specific composition belongs to the client adapter"
        ) from error
    return _ensure_rgb_uint8_image(value, "observation/image")


def _standard_eef_delta_to_abs_eef_pose(
    pose_delta: np.ndarray,
    initial_pos: np.ndarray,
    initial_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode standard SE(3) delta actions into absolute EEF poses.

    The model output is the Cosmos action-manifold representation
    ``[translation_delta(3), rot6d_delta(6)]`` using the shared
    ``backward_framewise`` convention. The returned pose sequence drops the
    initial conditioning pose and aligns one absolute target with each predicted
    delta row.
    """

    pose_delta = np.asarray(pose_delta, dtype=np.float32)
    if pose_delta.ndim != 2 or pose_delta.shape[-1] != 9:
        raise ValueError(f"Expected standard EEF pose delta shape (T, 9), got {pose_delta.shape}")

    initial_pose = build_abs_pose_from_components(
        np.asarray(initial_pos, dtype=np.float32).reshape(1, 3),
        np.asarray(initial_quat_xyzw, dtype=np.float32).reshape(1, 4),
        "quat_xyzw",
    )[0]
    poses_abs = pose_rel_to_abs(
        pose_delta,
        rotation_format="rot6d",
        pose_convention="backward_framewise",
        initial_pose=initial_pose,
        normalize_rotation=True,
    )
    target_poses = poses_abs[1:]
    positions = np.asarray(target_poses[:, :3, 3], dtype=np.float32)
    quat_xyzw = np.asarray(
        convert_rotation(target_poses[:, :3, :3], "matrix", "quat_xyzw", normalize_matrix=True),
        dtype=np.float32,
    )
    return positions, quat_xyzw


def _convert_gripper_semantics(
    values: np.ndarray,
    source: str,
    target: str,
) -> np.ndarray:
    if source == target:
        return values
    if {source, target} == {"open_fraction", "close_fraction"}:
        return 1.0 - values
    raise ValueError(f"Unsupported gripper semantics conversion: {source!r} -> {target!r}")


def _pack_gripper_at_index(joints: np.ndarray, gripper: np.ndarray, index: int) -> np.ndarray:
    if joints.shape[:-1] != gripper.shape[:-1] or gripper.shape[-1] != 1:
        raise ValueError(f"Cannot combine joint shape {joints.shape} with gripper shape {gripper.shape}")
    if not 0 <= index <= joints.shape[-1]:
        raise ValueError(f"Invalid gripper index {index} for {joints.shape[-1]} joint channels")
    return np.concatenate((joints[..., :index], gripper, joints[..., index:]), axis=-1)


def _build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]  # value: [...], batch item: [1,...]
        else:
            data_batch[key] = [value]
    return data_batch


def _load_openpi_websocket_policy_server() -> type[Any]:
    try:
        from openpi_server.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError:
        try:
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoboLab WebSocket serving uses OpenPI's WebsocketPolicyServer. Install it with "
                "`uv sync --all-extras --group=cu130-train --group=policy-server`, "
                "or install the full Physical-Intelligence/openpi package."
            ) from exc
    return WebsocketPolicyServer


@dataclass(frozen=True)
class RobolabPolicyConfig:
    checkpoint_path: str
    manifest: ActionPolicyManifest
    dataset_source: DatasetSourceDescription
    decode_video: bool
    seed: int
    deterministic_seed: bool
    guidance: float
    num_steps: int
    shift: float

    @property
    def conditioning_fps(self) -> int:
        return self.manifest.policy_fps

    @property
    def resolution(self) -> str | None:
        return self.manifest.transform.resolution

    @property
    def action_chunk_size(self) -> int:
        return self.manifest.chunk_size

    @property
    def action_dim(self) -> int:
        return self.manifest.model_action_dim

    @property
    def joint_dof(self) -> int:
        return self.action_dim - 1

    @property
    def image_height(self) -> int:
        return self.manifest.observation.canvas_shape_hw[0]

    @property
    def image_width(self) -> int:
        return self.manifest.observation.canvas_shape_hw[1]

    @property
    def history_length(self) -> int:
        return self.manifest.conditioning.history_rows

    @property
    def use_state(self) -> bool:
        return self.manifest.conditioning.state_rows == 1

    @property
    def action_space(self) -> str:
        return self.manifest.model_action.codec


class RobolabServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str = _DEFAULT_DROID_POLICY_CHECKPOINT
    """Consolidated local safetensors checkpoint directory, registered checkpoint name, or DCP path with --allow-dcp-checkpoint."""
    hf_revision: str = _DEFAULT_HF_REVISION
    """Hugging Face revision used when --checkpoint-path is a supported public RoboLab policy repository."""
    allow_dcp_checkpoint: bool = False
    """If set, allow direct DCP/S3 checkpoint loading instead of requiring a consolidated safetensors export."""
    experiment: str | None = None
    """Experiment name for DCP checkpoints using module configs, e.g. droid_lerobot_8b_policy."""
    config_file: str | None = None
    """Optional config file forwarded to OmniSetup, e.g. <dcp>/model/config.json."""
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    """Hydra experiment overrides forwarded to OmniSetup for DCP checkpoint loading."""
    credential_path: str | None = None
    """Optional checkpoint object-store credential path for DCP/S3 loading."""
    policy_config: Path | None = None
    """Explicit action_policy YAML/TOML. Otherwise discovered from the run/export root."""
    dataset_source: str | None = None
    """Exact datasets[].name to serve. Required for multi-source policies; single-source policies auto-select."""

    port: int = 8000
    """WebSocket port to bind."""
    host: str = "0.0.0.0"
    """WebSocket host to bind."""
    decode_video: bool = False
    """If set, decode and return the predicted rollout video as a uint8 NumPy array."""
    guardrails: bool = False
    """Enable text/video guardrails. Disabled by default for action-only policy serving."""

    output_dir: Path | None = None
    """Output directory for OmniInference. Defaults to /tmp/cosmos3_action_server/robolab."""
    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler used by OmniInference."""

    seed: int = 0
    """Base generation seed used to initialize the request RNG."""
    deterministic_seed: bool = False
    """Use the same seed for every request. If false, advance a NumPy RNG seeded by --seed."""
    guidance: float = 3.0
    """Guidance scale for denoising."""
    num_steps: int = 4
    """Number of denoising steps."""
    shift: float = 5.0
    """UniPC sampler shift."""


def _build_policy_contract(config: RobolabPolicyConfig) -> dict[str, Any]:
    return config.manifest.client_contract(config.dataset_source.name)


class RobolabPolicyService:
    def __init__(self, args: RobolabServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")
        requested_checkpoint = args.checkpoint_path
        resolved_checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path, hf_revision=args.hf_revision)
        manifest = _resolve_policy_manifest(
            resolved_checkpoint_path,
            requested_checkpoint=requested_checkpoint,
            policy_config=args.policy_config,
        )
        dataset_source = manifest.resolve_dataset_source(args.dataset_source)
        args = args.model_copy(update={"checkpoint_path": resolved_checkpoint_path})
        _validate_checkpoint(args.checkpoint_path, allow_dcp_checkpoint=args.allow_dcp_checkpoint)
        get_domain_id(manifest.domain_name)
        maybe_init_distributed()

        setup_args = self._build_setup_args(args)
        log.info(
            f"[robolab-policy-server] loading model: checkpoint_path={setup_args.checkpoint_path!r} "
            f"config_file={setup_args.config_file!r} experiment={setup_args.experiment!r}"
        )
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args

        self.cfg = RobolabPolicyConfig(
            checkpoint_path=self.setup_args.checkpoint_path,
            manifest=manifest,
            dataset_source=dataset_source,
            decode_video=bool(args.decode_video),
            seed=int(args.seed),
            deterministic_seed=bool(args.deterministic_seed),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            shift=float(args.shift),
        )
        self._transform = self._build_transform(manifest)

        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.cfg.seed)
        log.info(
            f"[robolab-policy-server] ready profile={manifest.profile_id!r} domain={manifest.domain_name!r} "
            f"dataset_source={dataset_source.name!r} "
            f"resolution={self.cfg.resolution!r} robot={manifest.robot} action_space={self.cfg.action_space} "
            f"action_dim={self.cfg.action_dim} wire_dim={manifest.wire_action_dim} "
            f"chunk={self.cfg.action_chunk_size} history={self.cfg.history_length} use_state={self.cfg.use_state} "
            f"image={self.cfg.image_height}x{self.cfg.image_width} fps={self.cfg.conditioning_fps} "
            f"guidance={self.cfg.guidance} num_steps={self.cfg.num_steps} shift={self.cfg.shift} "
            f"seed={self.cfg.seed} deterministic_seed={self.cfg.deterministic_seed} "
            f"model_gripper={manifest.model_action.gripper.semantics} "
            f"wire_gripper={manifest.wire_action.gripper.semantics}"
        )

    def _build_setup_args(self, args: RobolabServerArgs) -> OmniSetupArgs:
        setup_overrides: dict[str, Any] = {
            "checkpoint_path": args.checkpoint_path,
            "output_dir": args.output_dir or _DEFAULT_ROBOLAB_OUTPUT_DIR,
            "sampler": args.sampler,
            "guardrails": args.guardrails,
        }
        if args.experiment is not None:
            setup_overrides["experiment"] = args.experiment
        if args.config_file is not None:
            setup_overrides["config_file"] = args.config_file
        if args.experiment_overrides:
            setup_overrides["experiment_overrides"] = list(args.experiment_overrides)
        if args.credential_path is not None:
            setup_overrides["credential_path"] = args.credential_path
        overrides = OmniSetupOverrides.model_validate(setup_overrides)
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        return disable_runtime_ema_for_frozen_config(setup_args)

    def _build_transform(self, manifest: ActionPolicyManifest) -> ActionTransformPipeline:
        model_max_action_dim = getattr(getattr(self.model, "config", None), "max_action_dim", None)
        if isinstance(model_max_action_dim, int) and model_max_action_dim != manifest.transform.max_action_dim:
            raise ValueError(
                f"Manifest/model max_action_dim mismatch: {manifest.transform.max_action_dim} vs {model_max_action_dim}"
            )
        model_config = getattr(self.model, "config", None)
        vlm_config = getattr(model_config, "vlm_config", None)
        tokenizer_config = getattr(vlm_config, "tokenizer", None)
        if tokenizer_config is None and isinstance(vlm_config, dict):
            tokenizer_config = vlm_config.get("tokenizer")
        transform = manifest.transform
        return ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=0.0,
            max_action_dim=transform.max_action_dim,
            action_channel_masking=transform.action_channel_masking,
            append_viewpoint_info=transform.append_viewpoint_info,
            append_duration_fps_timestamps=transform.append_duration_fps_timestamps,
            append_resolution_info=transform.append_resolution_info,
            append_idle_frames=transform.append_idle_frames,
            format_prompt_as_json=transform.format_prompt_as_json,
        )

    def _next_seed(self) -> int:
        if self.cfg.deterministic_seed:
            return self.cfg.seed
        return int(self._rng.integers(0, 2**31))

    def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        image = _extract_observation_image(obs)
        image_h = self.cfg.image_height
        image_w = self.cfg.image_width
        if image.shape[:2] != (image_h, image_w):
            image = _resize_rgb_uint8(image, (image_h, image_w))
        t_frames = self.cfg.action_chunk_size + 1
        video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)  # [3,T,H,W]
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)  # [3,H,W]

        use_state_rows = 1 if self.cfg.use_state else 0
        action = torch.zeros(
            (self.cfg.action_chunk_size + use_state_rows, self.cfg.action_dim),
            dtype=torch.float32,
        )  # [T,D]
        history_action: torch.Tensor | None = None
        num_history_rows = self.cfg.history_length - use_state_rows
        gripper_position = _ensure_gripper_array(obs["observation/gripper_position"])
        gripper_position = _convert_gripper_semantics(
            gripper_position,
            "close_fraction",
            self.cfg.manifest.model_action.gripper.semantics,
        )

        if self.cfg.action_space == "joint_position":
            joint_position = _ensure_2d_float_array(
                obs["observation/joint_position"], "observation/joint_position", self.cfg.joint_dof
            )
            if self.cfg.use_state:
                packed = _pack_gripper_at_index(
                    joint_position[-1:],
                    gripper_position[-1:],
                    self.cfg.manifest.model_action.gripper.index,
                )
                action[0] = torch.from_numpy(packed[0])  # [D]
            if num_history_rows > 0:
                if len(joint_position) < num_history_rows + 1:
                    raise ValueError("Not enough joint_position rows for requested history_length")
                history_np = _pack_gripper_at_index(
                    joint_position[-num_history_rows - 1 : -1],
                    gripper_position[-num_history_rows - 1 : -1],
                    self.cfg.manifest.model_action.gripper.index,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        elif self.cfg.action_space == "eef_delta":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            if self.cfg.use_state:
                rot6d = convert_rotation(eef_quat[-1], "quat_xyzw", "rot6d")
                action[0] = torch.from_numpy(np.concatenate((eef_pos[-1], rot6d, gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                if len(eef_pos) < num_history_rows + 1 or len(eef_quat) < num_history_rows + 1:
                    raise ValueError("Not enough eef_pos/eef_quat rows for requested history_length")
                poses_abs = build_abs_pose_from_components(eef_pos, eef_quat, "quat_xyzw")
                poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
                history_np = np.concatenate(
                    [poses_rel[-num_history_rows:], gripper_position[-num_history_rows:]],
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]
        else:
            raise ValueError(f"Unsupported model action codec {self.cfg.action_space!r}")

        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.long),  # []
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id(self.cfg.manifest.domain_name), dtype=torch.long),  # []
            "viewpoint": self.cfg.manifest.observation.viewpoint,
            "additional_view_description": self.cfg.dataset_source.view_description,
        }
        if history_action is not None:
            sample["history_action"] = history_action
        return self._transform(sample, self.cfg.resolution, action_normalizer=None)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        infer_start = time.perf_counter()
        sample = self._build_sample(obs)
        data_batch = _build_data_batch_from_sample(sample)
        seed = self._next_seed()
        log.info(
            f"[robolab-policy-server] inference_chunk_start seed={seed} "
            f"chunk={self.cfg.action_chunk_size} num_steps={self.cfg.num_steps} "
            f"shift={self.cfg.shift} guidance={self.cfg.guidance} prompt={data_batch['ai_caption'][0]!r}"
        )

        generate_start = time.perf_counter()
        with self._lock:
            with torch.inference_mode():
                samples = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )
        generate_elapsed = time.perf_counter() - generate_start

        # ``generate_samples_from_batch`` already externalizes the action with
        # the batch's ActionProcessingRecord (unpad + denormalize). Reapplying
        # ActionProcessor here would denormalize affine policies twice.
        action = samples["action"][0]
        if action.ndim != 2 or action.shape[-1] != self.cfg.action_dim:
            raise RuntimeError(
                "Model returned a non-externalized action: "
                f"expected [T,{self.cfg.action_dim}], got {tuple(action.shape)}"
            )
        action = action[self.cfg.history_length :]  # [T2,D]
        if action.shape[0] != self.cfg.action_chunk_size:
            raise RuntimeError(
                "Model returned an unexpected external action horizon after conditioning rows: "
                f"expected {self.cfg.action_chunk_size}, got {action.shape[0]}"
            )
        action_np = action.detach().cpu().numpy()  # [T2,D]
        model_gripper_index = self.cfg.manifest.model_action.gripper.index
        action_np[:, model_gripper_index] = _convert_gripper_semantics(
            action_np[:, model_gripper_index],
            self.cfg.manifest.model_action.gripper.semantics,
            self.cfg.manifest.wire_action.gripper.semantics,
        )

        if self.cfg.action_space == "eef_delta":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            position, quat_xyzw = _standard_eef_delta_to_abs_eef_pose(
                action_np[:, :9],
                eef_pos[-1],
                eef_quat[-1],
            )
            gripper = action_np[:, model_gripper_index : model_gripper_index + 1]
            action_np = np.concatenate([position, quat_xyzw, gripper], axis=-1)

        outputs: dict[str, Any] = {"action": action_np}
        if self.cfg.decode_video:
            pred_vision_latent = samples["vision"][0]  # [C,T,H,W]
            video = self.model.decode(pred_vision_latent)  # [1,C,T,H,W]
            video = ((video[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)  # [T,H,W,3]
            outputs["video"] = video.detach().cpu().numpy()
        log.info(
            f"[robolab-policy-server] inference_chunk_end seed={seed} "
            f"elapsed_s={time.perf_counter() - infer_start:.3f} generate_s={generate_elapsed:.3f} "
            f"action_shape={tuple(action_np.shape)}"
        )
        return outputs


def serve(args: RobolabServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[robolab-policy-server] starting host={hostname} bind={args.host}:{int(args.port)}")
    service = RobolabPolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[robolab-policy-server] Server accessible at: ws://{local_ip}:{int(args.port)}/")
    log.info(f"[robolab-policy-server] Health check: http://{local_ip}:{int(args.port)}/healthz")
    server_cls = _load_openpi_websocket_policy_server()
    metadata = {"policy_contract": _build_policy_contract(service.cfg)}
    log.info(f"[robolab-policy-server] policy_contract={metadata['policy_contract']}")
    server_cls(policy=service, host=args.host, port=int(args.port), metadata=metadata).serve_forever()


def main() -> None:
    cascade_subcommand_args = getattr(
        tyro.conf,
        "CascadeSubcommandArgs",
        tyro.conf.ConsolidateSubcommandArgs,
    )
    args = tyro_cli(
        RobolabServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            cascade_subcommand_args,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
