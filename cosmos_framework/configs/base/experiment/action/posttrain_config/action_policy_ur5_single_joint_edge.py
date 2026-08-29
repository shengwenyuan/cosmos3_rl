# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Source-neutral Cosmos3-Edge post-training for stateful 7-D UR5 policies."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.ur5_single_lerobot_dataset import (
    get_action_ur5_single_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _edge_model_config() -> dict:
    config = copy.deepcopy(EDGE_MODEL_CONFIG)
    config["max_num_tokens_after_packing"] = -1
    config["activation_checkpointing"]["mode"] = "full"
    config["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    config["tokenizer"]["encode_exact_durations"] = [17]
    config["vlm_config"]["tokenizer"] = L(build_processor_lazy)(
        tokenizer_type="${oc.env:EDGE_BASE_PATH}"
    )
    return config


action_policy_ur5_single_joint_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_ur5_joint",
            group="stage2_c32",
            name="action_policy_ur5_single_joint_edge",
            wandb_mode="disabled",
        ),
        model=dict(config=_edge_model_config()),
        requires_action_policy_manifest=True,
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
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[23500],
            f_max=[1.0],
            f_min=[0.1],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=8,
            logging_iter=10,
            max_iter=23500,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=500,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=20),
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
            save_iter=500,
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
            dataset_name="action_ur5_single_joint_edge",
            max_samples_per_batch=1,
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
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    ur5_single=dict(
                        ratio=1,
                        dataset=L(get_action_ur5_single_sft_dataset)(
                            sources=[],
                            fps=15.0,
                            chunk_length=32,
                            video_subsample=2,
                            sample_stride=1,
                            split="full",
                            mode="wam",
                            viewpoint="concat_view",
                            action_normalization="quantile",
                            action_stats_path="${oc.env:ACTION_STATS_PATH}",
                            apply_forward_clamp=False,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            action_channel_masking=True,
                            append_viewpoint_info=True,
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            append_idle_frames=False,
                            format_prompt_as_json=True,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_ur5_single_joint_edge",
    node=action_policy_ur5_single_joint_edge,
)

# The fixed-subset gate reuses the same source-neutral model and loader. Only
# generic optimization-side sampling changes; episode identities remain owned
# by the TOML action-policy manifest.
action_policy_ur5_single_joint_edge_overfit = copy.deepcopy(action_policy_ur5_single_joint_edge)
action_policy_ur5_single_joint_edge_overfit.job.name = "action_policy_ur5_single_joint_edge_overfit"
_overfit_dataset = (
    action_policy_ur5_single_joint_edge_overfit.dataloader_train.dataloader.datasets.ur5_single.dataset
)
_overfit_dataset.sample_stride = 8
_overfit_dataset.cfg_dropout_rate = 0.0

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_ur5_single_joint_edge_overfit",
    node=action_policy_ur5_single_joint_edge_overfit,
)
