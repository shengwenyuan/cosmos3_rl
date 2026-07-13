# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

_INSTITUTIONS = [
    "AUTOLab",
    "CLVR",
    "GuptaLab",
    "ILIAD",
    "IPRL",
    "IRIS",
    "PennPAL",
    "RAD",
    "RAIL",
    "REAL",
    "RPL",
    "TRI",
    "WEIRD",
]

COSMOS3_DROID_SUCCESS_PROFILE = "cosmos3_droid_success_640x360_v1"

LEROBOT_ROOTS = {
    # Public Cosmos3-DROID is already rooted at its LeRobot ``success`` split.
    COSMOS3_DROID_SUCCESS_PROFILE: None,
    "droid_lerobot_20260115_no_noops": None,
    "droid_plus_lerobot_320x180_20260406_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_320x180_20260406": ["success", "failure"],
    "droid_plus_lerobot_640x360_20260412_sharded": [f"success/{x}" for x in _INSTITUTIONS]
    + [f"failure/{x}" for x in _INSTITUTIONS],
    "droid_plus_lerobot_640x360_20260412": ["success", "failure"],
}

IMAGE_FEATURES = {
    COSMOS3_DROID_SUCCESS_PROFILE: {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_lerobot_20260115_no_noops": {
        "wrist": "observation.images.wrist_image_left",
        "left": "observation.images.exterior_image_1_left",
        "right": "observation.images.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_320x180_20260406": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412_sharded": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
    "droid_plus_lerobot_640x360_20260412": {
        "wrist": "observation.image.wrist_image_left",
        "left": "observation.image.exterior_image_1_left",
        "right": "observation.image.exterior_image_2_left",
    },
}

STATE_FEATURES = {
    COSMOS3_DROID_SUCCESS_PROFILE: "observation.state.cartesian_position",
    "droid_lerobot_20260115_no_noops": "observation.state",
    "droid_plus_lerobot_320x180_20260406_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_320x180_20260406": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "observation.state.cartesian_position",
    "droid_plus_lerobot_640x360_20260412": "observation.state.cartesian_position",
}

ACTION_FEATURES = {
    COSMOS3_DROID_SUCCESS_PROFILE: "action.gripper_position",
    "droid_lerobot_20260115_no_noops": "action",
    "droid_plus_lerobot_320x180_20260406_sharded": "action.gripper_position",
    "droid_plus_lerobot_320x180_20260406": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412_sharded": "action.gripper_position",
    "droid_plus_lerobot_640x360_20260412": "action.gripper_position",
}

IS_FLAT_ACTION = {
    COSMOS3_DROID_SUCCESS_PROFILE: False,
    "droid_lerobot_20260115_no_noops": True,
    "droid_plus_lerobot_320x180_20260406_sharded": False,
    "droid_plus_lerobot_320x180_20260406": False,
    "droid_plus_lerobot_640x360_20260412_sharded": False,
    "droid_plus_lerobot_640x360_20260412": False,
}

HAS_MULTI_LANGUAGE_ANNOTATIONS = {
    COSMOS3_DROID_SUCCESS_PROFILE: True,
    "droid_lerobot_20260115_no_noops": False,
    "droid_plus_lerobot_320x180_20260406_sharded": True,
    "droid_plus_lerobot_320x180_20260406": True,
    "droid_plus_lerobot_640x360_20260412_sharded": True,
    "droid_plus_lerobot_640x360_20260412": True,
}

# Stored semantics are explicit; the released DROID model convention is
# open_fraction (0=closed, 1=open).
SOURCE_GRIPPER_SEMANTICS = {
    COSMOS3_DROID_SUCCESS_PROFILE: "close_fraction",
    "droid_lerobot_20260115_no_noops": "open_fraction",
    "droid_plus_lerobot_320x180_20260406_sharded": "close_fraction",
    "droid_plus_lerobot_320x180_20260406": "close_fraction",
    "droid_plus_lerobot_640x360_20260412_sharded": "close_fraction",
    "droid_plus_lerobot_640x360_20260412": "close_fraction",
}

_JOINT_ACTION_FEATURE = "action.joint_position"
_JOINT_STATE_FEATURE = "observation.state.joint_positions"
_GRIPPER_STATE_FEATURE = "observation.state.gripper_position"
