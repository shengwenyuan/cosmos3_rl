#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Total 64-GPU full training; PyTorchJob supplies node topology.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOML_FILE="$SCRIPT_DIR/toml/sft_config/action_policy_rh20t_ur5_joint_edge_64gpu.toml"
EXPECTED_GPU_WORLD_SIZE=64

source "$SCRIPT_DIR/_launch_sft_action_policy_rh20t_ur5_joint_edge.sh"

