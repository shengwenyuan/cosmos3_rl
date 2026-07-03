# UR5 post-training - local addition, not part of upstream Cosmos3.

"""``action_policy_berkeley_ur5_eef_nano`` - Berkeley AUTOLab UR5 EEF-space policy SFT.

Post-trains Cosmos3-Nano on the Berkeley AUTOLab UR5 LeRobot dataset as a 10D
EEF delta policy: ``[translation_delta(3), rot6d_delta(6), gripper(1)]``. This
is intentionally separate from the RoboMIND UR joint-space recipe.

Usage (1 node, 8 GPU)::

    BERKELEY_UR5_ROOT=/mlp_vepfs/share/swy/cosmos3-framework/lerobot/berkeley_autolab_ur5 \
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> WAN_VAE_PATH=<Wan2.2_VAE.pth> \
    torchrun --nproc_per_node=8 -m cosmos_framework.scripts.train \
        --sft-toml examples/toml/sft_config/action_policy_berkeley_ur5_eef_repro.toml
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.action.datasets.berkeley_ur5_eef_dataset import get_action_berkeley_ur5_eef_sft_dataset
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


action_policy_berkeley_ur5_eef_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
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
            name="action_policy_berkeley_ur5_eef_nano",
            wandb_mode="disabled",
        ),
        model=dict(config=copy.deepcopy(NANO_MODEL_CONFIG)),
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
            lr=1.0e-04,  # half of the joint-space full-batch setting before batch-size scaling
            lr_multipliers={
                "action2llm": 2.0,
                "llm2action": 2.0,
                "action_modality_embed": 2.0,
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
            dataset_name="action_berkeley_ur5_eef",
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
                datasets=dict(
                    berkeley_ur5_eef=dict(
                        ratio=1,
                        dataset=L(get_action_berkeley_ur5_eef_sft_dataset)(
                            root="${oc.env:BERKELEY_UR5_ROOT}",
                            fps=None,
                            chunk_length=32,
                            mode="policy",
                            gripper_invert=False,
                            canvas_views=("observation.images.image", "observation.images.hand_image"),
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            action_normalization=None,
                            viewpoint="concat_view",
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
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

action_policy_berkeley_ur5_eef_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]
action_policy_berkeley_ur5_eef_nano["model"]["config"]["max_num_tokens_after_packing"] = -1
action_policy_berkeley_ur5_eef_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_berkeley_ur5_eef_nano",
    node=action_policy_berkeley_ur5_eef_nano,
)
