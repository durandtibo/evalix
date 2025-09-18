from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from evalix.metric import mean_squared_log_error

############################################
#     Tests for mean_squared_log_error     #
############################################


def test_mean_squared_log_error_correct() -> None:
    assert objects_are_equal(
        mean_squared_log_error(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_correct_2d() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_incorrect() -> None:
    assert objects_are_allclose(
        mean_squared_log_error(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "mean_squared_log_error": 0.46117484006431314},
    )


def test_mean_squared_log_error_empty() -> None:
    assert objects_are_equal(
        mean_squared_log_error(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "mean_squared_log_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_log_error_prefix_suffix() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_mean_squared_log_error_suffix": 0.0},
    )


def test_mean_squared_log_error_nan_omit() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 3, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_nan_omit_y_true() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_nan_propagate() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "mean_squared_log_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_log_error_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="propagate",
        ),
        {"count": 6, "mean_squared_log_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_log_error_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "mean_squared_log_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_log_error_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        mean_squared_log_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


def test_mean_squared_log_error_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="raise",
        )


def test_mean_squared_log_error_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match=r"'y_pred' contains at least one NaN value"):
        mean_squared_log_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="raise",
        )
