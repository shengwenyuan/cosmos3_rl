# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Canvas helpers shared by action-policy LeRobot datasets."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def zero_like_view(reference: torch.Tensor) -> torch.Tensor:
    """Return a black view with the same shape, dtype, and device as ``reference``."""
    return torch.zeros_like(reference)


def concat_three_view_canvas(top: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Build the fixed DROID/Cosmos three-view canvas.

    The first view stays full resolution as the top row. The second and third
    views are resized to half height/width and concatenated side by side as the
    bottom row. Missing views should be passed in as black tensors matching the
    real view shape before calling this helper.
    """
    _, _, h, w = top.shape
    half_h, half_w = h // 2, w // 2
    left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
    right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
    return torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
