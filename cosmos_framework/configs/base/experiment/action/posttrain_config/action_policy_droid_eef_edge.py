# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge DROID Stage-1 EEF policy post-training.

The 4/8/16/32/64-GPU TOMLs own only topology and run-level scalars. This
experiment owns the shared Edge model, exact F3 window manifest adapter,
F2-compatible SE(3) smoothing, and train-only action normalization binding.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_droid_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

_EEF_LAYOUT = (
    "delta_x", "delta_y", "delta_z",
    "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
)
_DROID_VIEW_DESCRIPTION = (
    "The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated "
    "third-person perspective views of the scene from opposite sides, with the robot visible."
)


def _droid_eef_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["max_num_tokens_after_packing"] = -1
    cfg["activation_checkpointing"]["mode"] = "full"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["tokenizer"]["encode_exact_durations"] = [33]
    cfg["vlm_config"]["tokenizer"] = L(build_processor_lazy)(tokenizer_type="${oc.env:EDGE_BASE_PATH}")
    return cfg


action_policy_droid_eef_edge = LazyDict(
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
            project="cosmos3_action_droid_eef",
            group="stage1_c32",
            name="action_policy_droid_eef_edge",
            wandb_mode="disabled",
        ),
        model=dict(config=_droid_eef_edge_model_config()),
        requires_action_policy_manifest=True,
        action_policy=dict(
            schema_version=1,
            profile_id="droid_eef_stage1_15hz_c32",
            robot="droid",
            domain_name="droid_lerobot",
            policy_fps=15,
            chunk_size=32,
            model_action=dict(
                codec="eef_delta",
                layout=_EEF_LAYOUT,
                representation="delta",
                frame="droid_eef",
                pose_convention="backward_anchored",
                gripper=dict(index=9, semantics="open_fraction"),
            ),
            wire_action=dict(
                codec="eef_absolute",
                layout=("x", "y", "z", "qx", "qy", "qz", "qw", "gripper"),
                representation="absolute",
                frame="droid_eef",
                quaternion_order="xyzw",
                gripper=dict(index=7, semantics="close_fraction"),
            ),
            conditioning=dict(
                state_rows=0,
                history_rows=0,
                source="none",
                timing="32 future poses are encoded as SE(3) deltas anchored at the current EEF pose",
            ),
            decoder_anchor=dict(kind="current_eef_pose", frame="droid_eef", quaternion_order="xyzw"),
            observation=dict(
                layout_id="primary_top_aux_bottom_pair",
                view_shape_hw=(360, 640),
                canvas_shape_hw=(540, 640),
                view_roles=("primary", "aux_left", "aux_right"),
                missing_view_policy="error",
                viewpoint="concat_view",
                description="The wrist view is above the two exterior views.",
            ),
            transform=dict(
                resolution="480",
                max_action_dim=64,
                action_channel_masking=True,
                append_viewpoint_info=True,
                append_duration_fps_timestamps=True,
                append_resolution_info=True,
                append_idle_frames=False,
                format_prompt_as_json=True,
            ),
            normalization=dict(
                kind="quantile_rot",
                stats_file="artifacts/fx4_droid_franka_eef_c32_v2_quantile_rot.json",
                sha256="1965bc2a279ef9b7c82330f1102681d2d74a566ce655f5f1449a27647a6fec9a",
                apply_forward_clamp=False,
            ),
            datasets=(
                dict(
                    name="cosmos3_droid_success_640x360_v1",
                    root="${oc.env:DROID_ROOT}",
                    condition_source="none",
                    action_features=("observation.state.cartesian_position", "action.gripper_position"),
                    camera_features=dict(
                        primary="observation.image.wrist_image_left",
                        aux_left="observation.image.exterior_image_1_left",
                        aux_right="observation.image.exterior_image_2_left",
                    ),
                    action_layout=_EEF_LAYOUT,
                    gripper_semantics="close_fraction",
                    source_frame="droid_eef",
                    source_gripper_index=9,
                    source_target_offset=1,
                    description="DROID success trajectories filtered by Fx4 and resampled at 15 Hz.",
                    view_description=_DROID_VIEW_DESCRIPTION,
                ),
            ),
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
            lr=1.0e-05,
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
            cycle_lengths=[8000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[500],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=10,
            max_iter=8000,
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
            dataset_name="action_droid_eef_edge",
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
                    droid_eef=dict(
                        ratio=1,
                        dataset=L(get_action_droid_sft_dataset)(
                            root="${oc.env:DROID_ROOT}",
                            fps=15.0,
                            chunk_length=32,
                            action_space="ee_pose_delta",
                            mode="wam",
                            use_state=False,
                            split="train",
                            training_manifest_path="${oc.env:FX4_TRAIN_MANIFEST}",
                            pose_smoothing_window=5,
                            action_normalization="quantile_rot",
                            action_stats_path="${oc.env:ACTION_STATS_PATH}",
                            apply_forward_clamp=False,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            use_image_augmentation=True,
                            use_filter_dict=False,
                            dataset_profile="cosmos3_droid_success_640x360_v1",
                            viewpoint="concat_view",
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            action_channel_masking=True,
                            append_viewpoint_info=True,
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            append_idle_frames=False,
                            format_prompt_as_json=True,
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
    name="action_policy_droid_eef_edge",
    node=action_policy_droid_eef_edge,
)
