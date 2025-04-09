from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from evalix.metric import regression_errors

#######################################
#     Tests for regression_errors     #
#######################################


def test_regression_errors_correct() -> None:
    assert objects_are_equal(
        regression_errors(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {
            "count": 5,
            "mean_absolute_error": 0.0,
            "median_absolute_error": 0.0,
            "mean_squared_error": 0.0,
        },
    )


def test_regression_errors_correct_2d() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {
            "count": 6,
            "mean_absolute_error": 0.0,
            "median_absolute_error": 0.0,
            "mean_squared_error": 0.0,
        },
    )


def test_regression_errors_incorrect() -> None:
    assert objects_are_equal(
        regression_errors(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {
            "count": 4,
            "mean_absolute_error": 2.0,
            "median_absolute_error": 2.0,
            "mean_squared_error": 5.0,
        },
    )


def test_regression_errors_empty() -> None:
    assert objects_are_equal(
        regression_errors(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "mean_absolute_error": float("nan"),
            "median_absolute_error": float("nan"),
            "mean_squared_error": float("nan"),
        },
        equal_nan=True,
    )


def test_regression_errors_prefix_suffix() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_mean_absolute_error_suffix": 0.0,
            "prefix_median_absolute_error_suffix": 0.0,
            "prefix_mean_squared_error_suffix": 0.0,
        },
    )


def test_regression_errors_nan_omit() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {
            "count": 3,
            "mean_absolute_error": 0.0,
            "median_absolute_error": 0.0,
            "mean_squared_error": 0.0,
        },
    )


def test_regression_errors_nan_omit_y_true() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {
            "count": 5,
            "mean_absolute_error": 0.0,
            "median_absolute_error": 0.0,
            "mean_squared_error": 0.0,
        },
    )


def test_regression_errors_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {
            "count": 5,
            "mean_absolute_error": 0.0,
            "median_absolute_error": 0.0,
            "mean_squared_error": 0.0,
        },
    )


def test_regression_errors_nan_propagate() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="propagate",
        ),
        {
            "count": 6,
            "mean_absolute_error": float("nan"),
            "median_absolute_error": float("nan"),
            "mean_squared_error": float("nan"),
        },
        equal_nan=True,
    )


def test_regression_errors_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="propagate",
        ),
        {
            "count": 6,
            "mean_absolute_error": float("nan"),
            "median_absolute_error": float("nan"),
            "mean_squared_error": float("nan"),
        },
        equal_nan=True,
    )


def test_regression_errors_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="propagate",
        ),
        {
            "count": 6,
            "mean_absolute_error": float("nan"),
            "median_absolute_error": float("nan"),
            "mean_squared_error": float("nan"),
        },
        equal_nan=True,
    )


def test_regression_errors_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        regression_errors(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


def test_regression_errors_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="raise",
        )


def test_regression_errors_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        regression_errors(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="raise",
        )
