from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from evalix.metric import (
    binary_roc_auc,
    multiclass_roc_auc,
    multilabel_roc_auc,
    roc_auc,
)

#############################
#     Tests for roc_auc     #
#############################


def test_roc_auc_auto_binary() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_binary() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_roc_auc_suffix": 1.0},
    )


def test_roc_auc_binary_nan_omit() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            label_type="binary",
            nan_policy="omit",
        ),
        {"count": 3, "roc_auc": 1.0},
    )


def test_roc_auc_binary_nan_propagate() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": float("nan")},
        equal_nan=True,
    )


def test_roc_auc_binary_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            label_type="binary",
            nan_policy="raise",
        )


def test_roc_auc_auto_multiclass() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        {
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_multiclass() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_roc_multiclass_auc_nan_omit() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            label_type="multiclass",
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_multiclass_nan_propagate() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 7,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_roc_multiclass_auc_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            label_type="multiclass",
            nan_policy="raise",
        )


def test_roc_auc_auto_multilabel() -> None:
    assert objects_are_allclose(
        roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_multilabel() -> None:
    assert objects_are_allclose(
        roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_roc_auc_multilabel_nan_omit() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            label_type="multilabel",
            nan_policy="omit",
        ),
        {
            "count": 3,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_multilabel_nan_propagate() -> None:
    assert objects_are_equal(
        roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_roc_multilabel_auc_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            label_type="multilabel",
            nan_policy="raise",
        )


def test_roc_auc_label_type_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'label_type': incorrect"):
        roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="incorrect",
        )


####################################
#     Tests for binary_roc_auc     #
####################################


def test_binary_roc_auc_correct() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_binary_roc_auc_correct_2d() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([[1, 0, 0], [1, 1, 0]]),
            y_score=np.array([[2, -1, 0], [3, 1, -2]]),
        ),
        {"count": 6, "roc_auc": 1.0},
    )


def test_binary_roc_auc_incorrect() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1]),
            y_score=np.array([-1, 1, 0, -2]),
        ),
        {"count": 4, "roc_auc": 0.0},
    )


def test_binary_roc_auc_empty() -> None:
    assert objects_are_equal(
        binary_roc_auc(y_true=np.array([]), y_score=np.array([])),
        {"count": 0, "roc_auc": float("nan")},
        equal_nan=True,
    )


def test_binary_roc_auc_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_roc_auc_suffix": 1.0},
    )


def test_binary_roc_auc_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes:"):
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1, 6]),
        )


def test_binary_roc_auc_nan_omit() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 3, "roc_auc": 1.0},
    )


def test_binary_roc_auc_nan_omit_y_true() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 4, "roc_auc": 1.0},
    )


def test_binary_roc_auc_nan_omit_y_score() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 4, "roc_auc": 1.0},
    )


def test_binary_roc_auc_nan_propagate() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
        ),
        {"count": 5, "roc_auc": float("nan")},
        equal_nan=True,
    )


def test_binary_roc_auc_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {"count": 5, "roc_auc": float("nan")},
        equal_nan=True,
    )


def test_binary_roc_auc_nan_propagate_y_score() -> None:
    assert objects_are_equal(
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
        ),
        {"count": 5, "roc_auc": float("nan")},
        equal_nan=True,
    )


def test_binary_roc_auc_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="raise",
        )


def test_binary_roc_auc_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1]),
            nan_policy="raise",
        )


def test_binary_roc_auc_nan_raise_y_score() -> None:
    with pytest.raises(ValueError, match="'y_score' contains at least one NaN value"):
        binary_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="raise",
        )


########################################
#     Tests for multiclass_roc_auc     #
########################################


def test_multiclass_roc_auc_correct() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        {
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multiclass_roc_auc_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.3, 0.5],
                    [0.4, 0.4, 0.2],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        {
            "count": 6,
            "macro_roc_auc": 0.8333333333333334,
            "micro_roc_auc": 0.826388888888889,
            "roc_auc": np.array([0.9375, 0.8125, 0.75]),
            "weighted_roc_auc": 0.8333333333333334,
        },
    )


def test_multiclass_roc_auc_empty_1d() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(y_true=np.array([]), y_score=np.array([])),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_roc_auc_empty_2d() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(y_true=np.ones((0,)), y_score=np.ones((0, 3))),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_roc_auc_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_multiclass_roc_auc_nan_omit() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multiclass_roc_auc_nan_omit_y_true() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1],
                ]
            ),
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multiclass_roc_auc_nan_omit_y_score() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.1, 0.2, 0.7],
                ]
            ),
            nan_policy="omit",
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multiclass_roc_auc_nan_propagate() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
        ),
        {
            "count": 7,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_roc_auc_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1],
                ]
            ),
        ),
        {
            "count": 7,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_roc_auc_nan_propagate_y_score() -> None:
    assert objects_are_equal(
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
        ),
        {
            "count": 7,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_roc_auc_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            nan_policy="raise",
        )


def test_multiclass_roc_auc_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1],
                ]
            ),
            nan_policy="raise",
        )


def test_multiclass_roc_auc_nan_raise_y_score() -> None:
    with pytest.raises(ValueError, match="'y_score' contains at least one NaN value"):
        multiclass_roc_auc(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            nan_policy="raise",
        )


########################################
#     Tests for multilabel_roc_auc     #
########################################


def test_multilabel_roc_auc_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multilabel_roc_auc_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_score=np.array([[2], [-1], [0], [3], [1]]),
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multilabel_roc_auc_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
        ),
        {
            "count": 5,
            "macro_roc_auc": 0.6666666666666666,
            "micro_roc_auc": 0.5446428571428571,
            "roc_auc": np.array([1.0, 1.0, 0.0]),
            "weighted_roc_auc": 0.625,
        },
    )


def test_multilabel_roc_auc_empty_1d() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(y_true=np.array([]), y_score=np.array([])),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_roc_auc_empty_2d() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(y_true=np.ones((0, 3)), y_score=np.ones((0, 3))),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_roc_auc_prefix_suffix() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_multilabel_roc_auc_nan_omit() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            nan_policy="omit",
        ),
        {
            "count": 3,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multilabel_roc_auc_nan_omit_y_true() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multilabel_roc_auc_nan_omit_y_score() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_multilabel_roc_auc_nan_propagate() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
        ),
        {
            "count": 5,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_roc_auc_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
        ),
        {
            "count": 5,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_roc_auc_nan_propagate_y_score() -> None:
    assert objects_are_equal(
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
        ),
        {
            "count": 5,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_roc_auc_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            nan_policy="raise",
        )


def test_multilabel_roc_auc_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            nan_policy="raise",
        )


def test_multilabel_roc_auc_nan_raise_y_score() -> None:
    with pytest.raises(ValueError, match="'y_score' contains at least one NaN value"):
        multilabel_roc_auc(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array(
                [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
            ),
            nan_policy="raise",
        )
