# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets."""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "fractal": 20,
    
    # UR5e post-training — local additions. High ids (top of the 32-slot range; NVIDIA uses <=20)
    # leave 21..29 as a buffer against future upstream embodiments.
    "robomind-ur5-single": 30,  # single-arm UR5e, 7D joint_pos
    "robomind-ur5-dual": 31,  # dual-arm UR5e, 14D joint_pos
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    # NOTE: robomind-ur5-single/-dual are intentionally ABSENT here. This dict is the ee_pose /
    # midtrain (cartesian) raw width used only by offline FD/ID inference (inference/action.py)
    "embodiment_b": 30,
    "agibotworld": 29,
    "embodiment_c_gripper": 29,
    "embodiment_c_gripper_ext": 29,
    "fractal": 10,
    # NOTE: ``libero`` (7/10/13 depending on ``rotation_space``) and ``hand_pose``
    # (variable with ``keypoint_option`` and ``rotation_format``) are absent
    # because their raw width is set per-dataset at construction time. Inference
    # in inverse_dynamics/policy modes is not supported for these domains until
    # canonical widths are added here.
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def get_action_dim(embodiment_type: str) -> int:
    """Get the raw action dimension for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def is_valid_domain_name(embodiment_type: str) -> bool:
    """Check if the given embodiment type is recognized."""
    key = embodiment_type.lower().strip()
    return key in EMBODIMENT_TO_RAW_ACTION_DIM
