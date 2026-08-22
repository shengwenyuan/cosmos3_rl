# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset
from tools.droid.fx4_f2 import smooth_se3


def _bare_dataset() -> DROIDLeRobotDataset:
    dataset = DROIDLeRobotDataset.__new__(DROIDLeRobotDataset)
    dataset._chunk_length = 32
    dataset._split = "train"
    dataset._episode_records = []
    dataset._episode_cum_ends = []
    dataset._num_valid_indices = 0
    dataset._manifest_sample_rows = []
    dataset._episode_row_bounds = {}
    return dataset


def test_manifest_index_keeps_exact_noncontiguous_starts_in_episode_blocks() -> None:
    dataset = _bare_dataset()
    dataset._training_manifest = {
        0: {
            "episode_index": 0,
            "episode_id": "lab/success/episode-0",
            "split": "train",
            "kept_ranges": [
                {"selected_window_starts_c32": [0, 5, 12]},
                {"selected_window_starts_c32": [40]},
            ],
        }
    }
    meta = SimpleNamespace(
        episodes={
            "dataset_from_index": [100],
            "dataset_to_index": [180],
            "episode_id": ["lab/success/episode-0"],
        }
    )

    dataset._append_manifest_index_records(meta=meta, ds_idx=0)

    assert len(dataset) == 4
    assert dataset.get_shuffle_blocks() == [(0, 4)]
    assert [dataset._resolve_index(index)[1] for index in range(4)] == [100, 105, 112, 140]
    assert dataset._resolve_index(-1)[1] == 140


def test_context_smoothing_matches_f2_full_episode_values() -> None:
    dataset = _bare_dataset()
    dataset._training_manifest = {0: {}}
    dataset._pose_smoothing_window = 5
    dataset._state_features = "observation.state.cartesian_position"
    dataset._episode_row_bounds[(0, 0)] = (0, 50)

    steps = np.linspace(0.0, 1.0, 50)
    state = np.column_stack(
        (
            steps,
            0.1 * np.sin(steps * 5.0),
            0.05 * np.cos(steps * 3.0),
            0.2 * steps,
            -0.1 * steps,
            0.3 * steps,
        )
    )

    class _HF:
        def __getitem__(self, item):
            return {dataset._state_features: state[item]}

    dataset._get_dataset = lambda _index: SimpleNamespace(hf_dataset=_HF())
    expected_xyz, expected_rotation = smooth_se3(state[:, :3], Rotation.from_euler("xyz", state[:, 3:6]), 5)

    for row in (0, 5, 17):
        actual_xyz, actual_rotation = dataset._smoothed_eef_pose_window(0, row, 0)
        np.testing.assert_allclose(actual_xyz, expected_xyz[row : row + 33], atol=1e-12)
        np.testing.assert_allclose(
            actual_rotation.as_matrix(), expected_rotation[row : row + 33].as_matrix(), atol=1e-12
        )
