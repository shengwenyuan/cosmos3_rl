import numpy as np
from scipy.spatial.transform import Rotation

from tools.droid.eval_droid_eef_prediction_distribution import (
    absolute_pose_to_anchored_action,
    adjacent_motion,
    horizon_motion,
    matrix_to_rot6d,
    rot6d_to_matrix,
    select_indices,
    summarize,
)


def test_rot6d_round_trip() -> None:
    matrix = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(matrix)), matrix, atol=1e-12)


def test_absolute_pose_reconstructs_anchored_motion() -> None:
    initial_position = np.array([0.4, -0.2, 0.3])
    initial_rotation = Rotation.from_euler("xyz", [0.1, 0.2, -0.3])
    relative_position = np.array([[0.01, 0.02, -0.03], [0.04, -0.01, 0.02]])
    relative_rotation = Rotation.from_rotvec([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])
    positions = initial_position + initial_rotation.apply(relative_position)
    rotations = initial_rotation * relative_rotation
    action = absolute_pose_to_anchored_action(
        positions,
        rotations.as_quat(),
        initial_position,
        initial_rotation.as_quat(),
        np.array([0.2, 0.4]),
    )
    np.testing.assert_allclose(action[:, :3], relative_position, atol=1e-12)
    np.testing.assert_allclose(horizon_motion(action, (1, 2))[2][0], np.linalg.norm(relative_position[1]))
    step_translation, step_rotation = adjacent_motion(action)
    np.testing.assert_allclose(step_translation[0], np.linalg.norm(relative_position[0]))
    np.testing.assert_allclose(step_rotation, [0.1, 0.1], atol=1e-12)


def test_select_indices_spans_requested_shards() -> None:
    records = [(0, 0, 3, 10), (0, 0, 3, 11), (0, 0, 3, 12)]
    indices, shards, episodes = select_indices(
        records,
        [3, 6, 9],
        {10: (0, 0), 11: (0, 1), 12: (0, 2)},
        num_samples=6,
        num_shards=3,
        seed=0,
    )
    assert len(indices) == len(set(indices)) == 6
    assert len(shards) == 3
    assert set(episodes) == {10, 11, 12}


def test_summarize_passes_identical_prediction_and_target() -> None:
    action = np.zeros((2, 32, 10), dtype=np.float64)
    action[:, :, 0] = np.linspace(0.001, 0.032, 32)
    action[:, :, 3] = 1.0
    action[:, :, 7] = 1.0
    action[:, :, 9] = 0.5
    flat = action.reshape(-1, 10)
    stats = {
        "q01": (flat.min(axis=0) - 0.1).tolist(),
        "q99": (flat.max(axis=0) + 0.1).tolist(),
    }
    report = summarize(action, action, stats, ratio_limit=2.0, outlier_rate_limit=0.1)
    assert report["passed"]
    assert report["adjacent"]["translation_cm"]["ratio"]["p90"] == 1.0
