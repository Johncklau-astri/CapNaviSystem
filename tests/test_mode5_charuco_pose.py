import numpy as np

from capsule_calibration import _mode5_has_sufficient_board_coverage


def _board_points(x_values, y_values):
    return np.array(
        [[x, y, 0.0] for x in x_values for y in y_values],
        dtype=np.float64,
    ).reshape(-1, 1, 3)


def test_mode5_fallback_accepts_well_distributed_corners():
    points = _board_points([0.0, 0.04], [0.0, 0.02, 0.04]).copy()
    points = np.concatenate((points, _board_points([0.02], [0.0, 0.04])), axis=0)

    assert len(points) == 8
    assert _mode5_has_sufficient_board_coverage(points)


def test_mode5_fallback_rejects_clustered_corners():
    points = _board_points([0.02, 0.03, 0.04], [0.02, 0.03, 0.04])

    assert not _mode5_has_sufficient_board_coverage(points)