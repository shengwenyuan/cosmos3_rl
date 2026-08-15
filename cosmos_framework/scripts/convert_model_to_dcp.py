# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert a Hugging Face model to a DCP checkpoint."""

from cosmos_framework.inference.common.init import init_script

init_script(
    env={
        "COSMOS_DEVICE": "cpu",
    }
)

import copy
import math
import shutil
from pathlib import Path
from typing import Annotated

import pydantic
import torch
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.dcp import CustomSavePlanner
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import CheckpointOverrides, ResolvedPath
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.public_model_config import build_public_model_config
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

_AVAE_REGISTRY_URI = "s3://bucket/pretrained/tokenizers/audio/avae"


def _redirect_avae_to_local(hf_path):
    """Point the AVAE registry entry at hf_path/sound_tokenizer/.

    Pre-seeds the CheckpointDirHf._path cache so the registered AVAE checkpoint
    resolves to the local sibling directory instead of fetching the pinned
    revision of nvidia/Cosmos3-Nano from the HF Hub during hydra instantiation.
    """
    sound_tokenizer_dir = hf_path / "sound_tokenizer"
    if not sound_tokenizer_dir.is_dir():
        return
    register_checkpoints()
    avae = _CHECKPOINTS.get(_AVAE_REGISTRY_URI)
    if avae is not None:
        avae.hf._path = str(sound_tokenizer_dir)


def _build_runtime_model_config(model_dict: dict, hf_path: Path) -> dict:
    """Build a conversion-only config that keeps Edge tokenizers local.

    The registered Edge model config intentionally names the public Hub repo so
    normal inference can resolve its processor. During conversion from a full
    local snapshot, resolving that repo again is redundant and breaks offline
    conversion. The VAE is also not part of the model state written to DCP, so
    avoid constructing it solely for checkpoint conversion.

    Keep these changes runtime-only: the DCP's saved config remains the original
    deployment config rather than embedding a machine-specific local path.
    """

    runtime_model_dict = copy.deepcopy(model_dict)
    config = runtime_model_dict.get("config")
    if not isinstance(config, dict):
        return runtime_model_dict
    vlm_config = config.get("vlm_config")
    if not isinstance(vlm_config, dict):
        return runtime_model_dict
    tokenizer = vlm_config.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("repository") != "nvidia/Cosmos3-Edge":
        return runtime_model_dict

    tokenizer.pop("repository")
    tokenizer.pop("revision", None)
    tokenizer.pop("subdir", None)
    tokenizer["tokenizer_type"] = str(hf_path)
    config["load_vision_tokenizer"] = False
    return runtime_model_dict


class Args(pydantic.BaseModel):
    checkpoint: CheckpointOverrides
    """Hugging Face checkpoint."""
    output_path: Annotated[ResolvedPath, tyro.conf.arg(aliases=("-o",))]
    """Output DCP checkpoint directory."""


def convert_model_to_dcp(args: Args):
    print("Loading model...")
    checkpoint_config = args.checkpoint.build_checkpoint(checkpoints=OmniSetupOverrides.CHECKPOINTS)
    hf_path = checkpoint_config.download_checkpoint()
    _redirect_avae_to_local(hf_path)
    model_dict = checkpoint_config.load_model_config_dict()
    output_hf_config = Cosmos3OmniConfig(model=build_public_model_config(model_dict))
    runtime_model_dict = _build_runtime_model_config(model_dict, hf_path)
    runtime_hf_config = Cosmos3OmniConfig(model=build_public_model_config(runtime_model_dict))
    hf_model = Cosmos3OmniModel.from_pretrained_dcp(hf_path, config=runtime_hf_config)
    state_dict = get_model_state_dict(hf_model.model)

    # Match transformers default max shard size = 5GB.
    max_shard_size = 5 * 1024**3
    model_size = sum(p.numel() * p.element_size() for p in state_dict.values() if isinstance(p, torch.Tensor))
    thread_count = math.ceil(model_size / max_shard_size)

    print("Saving model...")
    storage_writer = FileSystemWriter(args.output_path / "model", thread_count=thread_count)
    dcp.save(state_dict=state_dict, storage_writer=storage_writer, planner=CustomSavePlanner())
    # ``checkpoint.json`` only exists for DCP-format source repos (e.g. Cosmos3-Nano);
    # safetensors/diffusers-layout repos (e.g. Cosmos3-Edge) don't ship it. Copy when present.
    source_checkpoint_json = hf_path / "checkpoint.json"
    if source_checkpoint_json.exists():
        shutil.copy(source_checkpoint_json, args.output_path / "checkpoint.json")
    output_hf_config.save_pretrained(args.output_path / "model")
    print(f"Saved checkpoint to {args.output_path}")


def main():
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    convert_model_to_dcp(args)


if __name__ == "__main__":
    main()
