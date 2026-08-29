#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# One-node fixed 16-episode overfit gate.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOML_FILE="$SCRIPT_DIR/toml/sft_config/action_policy_rh20t_ur5_joint_edge_8gpu_overfit.toml"
EXPECTED_GPU_WORLD_SIZE=8

source "$SCRIPT_DIR/_launch_sft_action_policy_rh20t_ur5_joint_edge.sh"

