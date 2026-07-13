# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared Cosmos3-Nano training stack for action-policy SFT recipes.

Concrete experiments own their dataset configuration and inject the resulting
lazy dataset node here. This module contains only the training stack shared by
DROID, Berkeley UR5 EEF, and the UR5 single/legacy recipes.
"""

import copy

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

_DEFAULT_CALLBACKS = (
    "basic",
    "optimization",
    "job_monitor",
    "training_stats",
)


def build_action_policy_nano_experiment(
    *,
    name: str,
    dataset_name: str,
    dataset_key: str,
    dataset_node: LazyDict,
    base_lr: float = 2.0e-04,
    action_head_lr_multiplier: float = 5.0,
    callbacks: tuple[str, ...] = _DEFAULT_CALLBACKS,
    requires_action_policy_manifest: bool = True,
) -> LazyDict:
    """Build the shared action-policy experiment around ``dataset_node``."""
    cfg = LazyDict(
        dict(
            defaults=[
                {"override /model": "mot_fsdp"},
                {"override /data_train": None},
                {"override /data_val": None},
                # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
                # diverged on the action loss).
                {"override /optimizer": "fusedadamw"},
                {"override /scheduler": "lambdalinear"},
                {"override /checkpoint": "s3"},
                {"override /callbacks": list(callbacks)},
                {"override /ema": "power"},
                {"override /tokenizer": "wan2pt2_tokenizer"},
                {"override /sound_tokenizer": None},
                {"override /vlm_config": None},
                {"override /ckpt_type": "dcp"},
                "_self_",
            ],
            job=dict(
                project="cosmos3",
                group="action_sft",
                name=name,
                wandb_mode="disabled",
            ),
            model=dict(
                config=copy.deepcopy(NANO_MODEL_CONFIG),
            ),
            optimizer=dict(
                betas=[0.9, 0.99],
                eps=1.0e-08,
                fused=True,
                keys_to_select=[
                    "moe_gen",
                    "time_embedder",
                    "vae2llm",
                    "llm2vae",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                ],
                lr=base_lr,
                lr_multipliers={
                    "action2llm": action_head_lr_multiplier,
                    "llm2action": action_head_lr_multiplier,
                    "action_modality_embed": action_head_lr_multiplier,
                },
                optimizer_type="FusedAdam",
                weight_decay=0.05,
            ),
            scheduler=dict(
                lr_scheduler_type="LambdaLinear",
                cycle_lengths=[100],
                f_max=[0.4],
                f_min=[0.0],
                f_start=[0.0],
                verbosity_interval=0,
                warm_up_steps=[0],
            ),
            trainer=dict(
                distributed_parallelism="fsdp",
                grad_accum_iter=1,
                logging_iter=1,
                max_iter=100,
                max_val_iter=None,
                run_validation=False,
                run_validation_on_start=False,
                save_zero_checkpoint=False,
                seed=42,
                timeout_period=999999999,
                validation_iter=100,
                compile_config=dict(recompile_limit=8, use_duck_shape=False),
                cudnn=dict(benchmark=True, deterministic=False),
                ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
                grad_scaler_args=dict(enabled=False),
                callbacks=dict(
                    dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                    device_monitor=dict(
                        every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                    ),
                    grad_clip=dict(clip_norm=1.0, force_finite=True),
                    heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                    iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                    low_precision=dict(update_iter=1),
                    manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                    param_count=dict(save_s3=False),
                    skip_nan_step=dict(max_consecutive_nan=100),
                    training_stats=dict(log_freq=100),
                ),
            ),
            checkpoint=dict(
                broadcast_via_filesystem=False,
                dcp_async_mode_enabled=False,
                enable_gcs_patch_in_boto3=True,
                keys_not_to_resume=[],
                keys_to_skip_loading=[
                    "net_ema.",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                    "action_pos_embed",
                ],
                load_ema_to_reg=False,
                load_path="???",
                load_training_state=False,
                only_load_scheduler_state=False,
                save_iter=100,
                strict_resume=False,
                verbose=True,
                hf_export=dict(
                    enabled=False,
                    export_every_n=1,
                    hf_repo_id=None,
                    upload_to_object_store=dict(bucket="", credentials="", enabled=False),
                ),
                jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
                load_from_object_store=dict(bucket="", credentials="", enabled=False),
                save_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            dataloader_train=L(PackingDataLoader)(
                audio_sample_rate=48000,
                dataset_name=dataset_name,
                max_samples_per_batch=128,
                max_sequence_length=None,
                patch_spatial=2,
                sound_latent_fps=0,
                tokenizer_spatial_compression_factor=16,
                tokenizer_temporal_compression_factor=4,
                dataloader=L(RankPartitionedDataLoader)(
                    batch_size=1,
                    in_order=False,
                    num_workers=4,
                    persistent_workers=True,
                    pin_memory=True,
                    prefetch_factor=4,
                    sampler=None,
                    datasets={
                        dataset_key: dict(
                            ratio=1,
                            dataset=dataset_node,
                        ),
                    },
                ),
            ),
            dataloader_val=None,
            requires_action_policy_manifest=requires_action_policy_manifest,
            upload_reproducible_setup=False,
        ),
        flags={"allow_objects": True},
    )

    # chunk_length=32 -> 33 observation frames; keep the model-side VAE duration aligned.
    cfg["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]
    cfg["model"]["config"]["max_num_tokens_after_packing"] = -1
    cfg["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0
    return cfg
