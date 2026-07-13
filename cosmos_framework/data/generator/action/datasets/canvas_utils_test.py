# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.canvas_utils import concat_three_view_canvas, resize_view

pytestmark = pytest.mark.level(0)


def test_explicit_view_resize_produces_declared_three_view_canvas() -> None:
    raw = torch.zeros((2, 3, 480, 640))
    view = resize_view(raw, (360, 640))
    canvas = concat_three_view_canvas(view, view, torch.zeros_like(view))

    assert view.shape == (2, 3, 360, 640)
    assert canvas.shape == (2, 3, 540, 640)
