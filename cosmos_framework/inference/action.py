# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.generator.action.action_processing import (
    ActionProcessingRecord,
    make_batched_action_processing_fields,
    pad_action_to_max_dim,
)
from cosmos_framework.data.generator.action.domain_utils import EMBODIMENT_TO_RAW_ACTION_DIM, get_domain_id
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.transforms import (
    build_sequence_plan_from_mode,
    find_closest_target_size,
    reflection_pad_to_target,
)
from cosmos_framework.inference.args import ModelMode
from cosmos_framework.inference.vision import read_media_frames
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

_MINE_MARKER = "MINE_div"


def _log_mine(stage: str, event: str, **fields: Any) -> None:
    field_text = " ".join(f"{key}={value!r}" for key, value in fields.items())
    suffix = f" {field_text}" if field_text else ""
    log.info(f"{_MINE_MARKER} stage={stage} event={event}{suffix}", rank0_only=False)


def _load_actions(
    action_path: Path | str | None,
    model_mode: ModelMode,
    action_chunk_size: int,
    max_action_dim: int,
    raw_action_dim: int | None,
) -> torch.Tensor:
    """Load actions from JSON (or zeros for policy mode and inverse dynamics mode). Returns padded action tensor."""
    total_start = time.perf_counter()
    _log_mine(
        "action_load",
        "start",
        action_path=str(action_path) if action_path is not None else None,
        model_mode=model_mode.value,
        action_chunk_size=action_chunk_size,
        max_action_dim=max_action_dim,
        raw_action_dim=raw_action_dim,
    )
    match model_mode:
        case ModelMode.FORWARD_DYNAMICS:
            assert action_path is not None, "action_path is required for forward_dynamics mode"
            p = Path(str(action_path))
            read_start = time.perf_counter()
            _log_mine("action_load", "json_read_start", action_path=str(p))
            raw_json = p.read_text()
            raw = torch.tensor(json.loads(raw_json), dtype=torch.float32)
            _log_mine(
                "action_load",
                "json_read_end",
                action_path=str(p),
                raw_shape=tuple(raw.shape),
                bytes=len(raw_json),
                elapsed_s=f"{time.perf_counter() - read_start:.3f}",
            )
            raw_dim = raw.shape[-1]
            assert raw_dim == raw_action_dim, (
                f"Raw action dimension from file ({raw_dim}) does not match expected dimension ({raw_action_dim})"
            )
            pad_start = time.perf_counter()
            padded = pad_action_to_max_dim(raw, max_action_dim)
            _log_mine(
                "action_load",
                "end",
                output_shape=tuple(padded.shape),
                pad_elapsed_s=f"{time.perf_counter() - pad_start:.3f}",
                elapsed_s=f"{time.perf_counter() - total_start:.3f}",
            )
            return padded
        case ModelMode.POLICY | ModelMode.INVERSE_DYNAMICS:
            assert raw_action_dim is not None, "raw_action_dim is required for policy and inverse_dynamics modes"
            action = torch.zeros(action_chunk_size, max_action_dim, dtype=torch.float32)
            _log_mine(
                "action_load",
                "zeros_end",
                output_shape=tuple(action.shape),
                elapsed_s=f"{time.perf_counter() - total_start:.3f}",
            )
            return action
        case _:
            raise ValueError(f"Unsupported action model_mode: {model_mode}")


def _format_prompt(
    prompt: str,
    view_point: str,
    video: torch.Tensor,
    action: torch.Tensor,
    fps: torch.Tensor,
    image_size: torch.Tensor,
) -> str:
    """Helper function to build the action prompt with optional duration and resolution info."""
    data_dict = {
        "viewpoint": view_point,
        "ai_caption": prompt.strip(),
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
    }
    prompt_json_formatter = ActionPromptJsonFormatter()
    ai_caption = prompt_json_formatter(data_dict)[prompt_json_formatter.caption_key]
    if isinstance(ai_caption, dict):
        ai_caption = json.dumps(ai_caption)
    return ai_caption


def build_action_batch(
    *,
    video: torch.Tensor,
    action: torch.Tensor,
    raw_action_dim: int,
    prompt: str,
    view_point: str,
    domain_name: str,
    model_mode: ModelMode,
    action_chunk_size: int,
    fps: int,
    resolution: str | None = None,
    input_video_key: str,
    batch_size: int = 1,
    device: Any = "cuda",
) -> dict:
    """Build an Action data batch from pre-loaded video and action tensors."""
    total_start = time.perf_counter()
    target_frames = action_chunk_size + 1
    _, num_frames, h, w = video.shape
    _log_mine(
        "action_batch",
        "start",
        model_mode=model_mode.value,
        domain_name=domain_name,
        input_video_key=input_video_key,
        video_shape=tuple(video.shape),
        action_shape=tuple(action.shape),
        raw_action_dim=raw_action_dim,
        action_chunk_size=action_chunk_size,
        fps=fps,
        resolution=resolution,
        batch_size=batch_size,
    )

    temporal_start = time.perf_counter()
    if num_frames < target_frames:
        pad = video[:, -1:].repeat(1, target_frames - num_frames, 1, 1)
        video = torch.cat([video, pad], dim=1)
    elif num_frames > target_frames:
        video = video[:, :target_frames]
    _log_mine(
        "action_batch",
        "temporal_fit_end",
        target_frames=target_frames,
        output_shape=tuple(video.shape),
        elapsed_s=f"{time.perf_counter() - temporal_start:.3f}",
    )

    if resolution is None:
        resolution = get_vision_data_resolution((h, w))

    target_w, target_h = find_closest_target_size(h, w, resolution)
    pad_dict: dict[str, Any] = {"video": video}
    pad_start = time.perf_counter()
    _log_mine(
        "action_batch",
        "reflection_pad_start",
        target_h=target_h,
        target_w=target_w,
        input_shape=tuple(video.shape),
    )
    reflection_pad_to_target(pad_dict, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h)
    video_padded = pad_dict["video"]
    padded_image_size = pad_dict["image_size"]
    _log_mine(
        "action_batch",
        "reflection_pad_end",
        output_shape=tuple(video_padded.shape),
        image_size=tuple(padded_image_size.shape),
        elapsed_s=f"{time.perf_counter() - pad_start:.3f}",
    )

    plan_start = time.perf_counter()
    _log_mine("action_batch", "sequence_plan_start", mode=model_mode.value)
    sequence_plan = build_sequence_plan_from_mode(
        mode=model_mode.value,
        video_length=target_frames,
        action_length=action_chunk_size,
        has_text=True,
    )
    _log_mine("action_batch", "sequence_plan_end", elapsed_s=f"{time.perf_counter() - plan_start:.3f}")

    prompt_start = time.perf_counter()
    _log_mine("action_batch", "format_prompt_start", prompt_chars=len(prompt), view_point=view_point)
    ai_caption = _format_prompt(
        prompt=prompt,
        view_point=view_point,
        video=video_padded,
        action=action,
        fps=torch.tensor(fps, dtype=torch.long),
        image_size=padded_image_size,
    )
    _log_mine(
        "action_batch",
        "format_prompt_end",
        ai_caption_chars=len(ai_caption),
        elapsed_s=f"{time.perf_counter() - prompt_start:.3f}",
    )

    action_processing_record = ActionProcessingRecord(
        raw_action_dim=raw_action_dim,
        action_normalizer=None,
    )

    batch = {
        input_video_key: [[video_padded]] * batch_size,
        "action": [[action]] * batch_size,
        **make_batched_action_processing_fields(action_processing_record, batch_size),
        "mode": [model_mode.value] * batch_size,
        "ai_caption": [ai_caption] * batch_size,
        "prompt": [prompt] * batch_size,
        "conditioning_fps": [torch.tensor(fps, dtype=torch.long)] * batch_size,
        "image_size": padded_image_size.unsqueeze(0).to(device=device),
        "domain_id": [torch.tensor(get_domain_id(domain_name), dtype=torch.long)] * batch_size,
        "sequence_plan": [sequence_plan] * batch_size,
    }
    _log_mine(
        "action_batch",
        "end",
        keys=sorted(batch.keys()),
        video_shape=tuple(video_padded.shape),
        action_shape=tuple(action.shape),
        elapsed_s=f"{time.perf_counter() - total_start:.3f}",
    )
    return batch


def get_action_sample_data(
    model_config: Any,
    *,
    batch_size: int,
    prompt: str,
    vision_path: Path,
    model_mode: ModelMode,
    action_path: Path | None,
    domain_name: str,
    view_point: str = "ego_view",
    resolution: str,
    action_chunk_size: int,
    max_action_dim: int,
    fps: int,
    device: Any,
) -> dict:
    """Load observation image/video + optional actions and build an Action inference batch."""
    total_start = time.perf_counter()
    domain_name = domain_name.lower().strip()
    _log_mine(
        "action_sample_data",
        "start",
        vision_path=str(vision_path),
        action_path=str(action_path) if action_path is not None else None,
        model_mode=model_mode.value,
        domain_name=domain_name,
        resolution=resolution,
        action_chunk_size=action_chunk_size,
        batch_size=batch_size,
    )
    if domain_name not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"invalid domain_name {domain_name!r}; expected one of {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )

    raw_action_dim = EMBODIMENT_TO_RAW_ACTION_DIM[domain_name]
    read_start = time.perf_counter()
    _log_mine("action_sample_data", "read_media_frames_start", vision_path=str(vision_path))
    frames, _ = read_media_frames(Path(vision_path), max_frames=action_chunk_size + 1)
    _log_mine(
        "action_sample_data",
        "read_media_frames_end",
        vision_path=str(vision_path),
        frames_shape=tuple(frames.shape),
        elapsed_s=f"{time.perf_counter() - read_start:.3f}",
    )
    assert action_path is not None or raw_action_dim is not None, (
        "Either action_path or raw_action_dim must be provided"
    )
    action_start = time.perf_counter()
    action = _load_actions(action_path, model_mode, action_chunk_size, max_action_dim, raw_action_dim)
    _log_mine(
        "action_sample_data",
        "load_actions_end",
        action_shape=tuple(action.shape),
        elapsed_s=f"{time.perf_counter() - action_start:.3f}",
    )

    batch_start = time.perf_counter()
    batch = build_action_batch(
        video=frames,
        action=action,
        raw_action_dim=raw_action_dim,
        prompt=prompt,
        view_point=view_point,
        domain_name=domain_name,
        model_mode=model_mode,
        action_chunk_size=action_chunk_size,
        fps=fps,
        resolution=resolution,
        input_video_key=model_config.input_video_key,
        batch_size=batch_size,
        device=device,
    )
    _log_mine(
        "action_sample_data",
        "end",
        keys=sorted(batch.keys()),
        build_batch_elapsed_s=f"{time.perf_counter() - batch_start:.3f}",
        elapsed_s=f"{time.perf_counter() - total_start:.3f}",
    )
    return batch
