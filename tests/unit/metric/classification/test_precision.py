from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from evalix.metric import (
    binary_precision,
    multiclass_precision,
    multilabel_precision,
    precision,
)
from evalix.metric.classification.precision import find_label_type

###############################
#     Tests for precision     #
###############################


def test_precision_auto_binary() -> None:
    assert objects_are_equal(
        precision(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "precision": 1.0},
    )


def test_precision_binary() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "precision": 1.0},
    )


def test_precision_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_precision_suffix": 1.0},
    )


def test_precision_binary_nan_omit() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            nan_policy="omit",
        ),
        {"count": 5, "precision": 1.0},
    )


def test_precision_binary_nan_propagate() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            nan_policy="propagate",
        ),
        {"count": 6, "precision": float("nan")},
        equal_nan=True,
    )


def test_precision_binary_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            nan_policy="raise",
        )


def test_precision_auto_multiclass() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_multiclass() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


def test_precision_multiclass_nan_omit() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_precision_multiclass_nan_propagate() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            nan_policy="propagate",
        ),
        {
            "count": 7,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_multiclass_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            nan_policy="raise",
        )


def test_precision_auto_multilabel() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_multilabel() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


def test_precision_multilabel_nan_omit() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            label_type="multilabel",
            nan_policy="omit",
        ),
        {
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_precision_multilabel_nan_propagate() -> None:
    assert objects_are_equal(
        precision(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            label_type="multilabel",
            nan_policy="propagate",
        ),
        {
            "count": 6,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_multilabel_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        precision(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            label_type="multilabel",
            nan_policy="raise",
        )


def test_precision_label_type_incorrect() -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'label_type': incorrect"):
        precision(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


######################################
#     Tests for binary_precision     #
######################################


def test_binary_precision_correct_1d() -> None:
    assert objects_are_equal(
        binary_precision(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "precision": 1.0},
    )


def test_binary_precision_correct_2d() -> None:
    assert objects_are_equal(
        binary_precision(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "precision": 1.0},
    )


def test_binary_precision_incorrect() -> None:
    assert objects_are_equal(
        binary_precision(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0])),
        {"count": 4, "precision": 0.5},
    )


def test_binary_precision_empty() -> None:
    assert objects_are_equal(
        binary_precision(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "precision": float("nan")},
        equal_nan=True,
    )


def test_binary_precision_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_precision_suffix": 1.0},
    )


def test_binary_precision_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match=r"'y_true' and 'y_pred' have different shapes:"):
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


def test_binary_precision_nan_omit() -> None:
    assert objects_are_allclose(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="omit",
        ),
        {"count": 4, "precision": 1.0},
    )


def test_binary_precision_omit_y_true() -> None:
    assert objects_are_allclose(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            nan_policy="omit",
        ),
        {"count": 5, "precision": 1.0},
    )


def test_binary_precision_omit_y_pred() -> None:
    assert objects_are_allclose(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="omit",
        ),
        {"count": 5, "precision": 1.0},
    )


def test_binary_precision_nan_propagate() -> None:
    assert objects_are_allclose(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 6, "precision": float("nan")},
        equal_nan=True,
    )


def test_binary_precision_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        ),
        {"count": 6, "precision": float("nan")},
        equal_nan=True,
    )


def test_binary_precision_nan_propagate_y_pred() -> None:
    assert objects_are_allclose(
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 6, "precision": float("nan")},
        equal_nan=True,
    )


def test_binary_precision_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="raise",
        )


def test_binary_precision_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            nan_policy="raise",
        )


def test_binary_precision_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match=r"'y_pred' contains at least one NaN value"):
        binary_precision(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="raise",
        )


##########################################
#     Tests for multiclass_precision     #
##########################################


def test_multiclass_precision_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_precision(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
        ),
        {
            "count": 6,
            "macro_precision": 0.5,
            "micro_precision": 0.6666666666666666,
            "precision": np.array([1.0, 0.5, 0.0]),
            "weighted_precision": 0.5,
        },
    )


def test_multiclass_precision_empty() -> None:
    assert objects_are_allclose(
        multiclass_precision(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_precision_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


def test_multiclass_precision_nan_omit() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="omit",
        ),
        {
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_omit_y_true() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_omit_y_pred() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_nan_propagate() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "count": 7,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_precision_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
        ),
        {
            "count": 7,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_precision_nan_propagate_y_pred() -> None:
    assert objects_are_allclose(
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "count": 7,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_precision_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="raise",
        )


def test_multiclass_precision_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            nan_policy="raise",
        )


def test_multiclass_precision_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match=r"'y_pred' contains at least one NaN value"):
        multiclass_precision(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="raise",
        )


##########################################
#     Tests for multilabel_precision     #
##########################################


def test_multilabel_precision_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_precision(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_precision(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "count": 5,
            "macro_precision": 0.6666666666666666,
            "micro_precision": 0.7142857142857143,
            "precision": np.array([1.0, 1.0, 0.0]),
            "weighted_precision": 0.625,
        },
    )


def test_multilabel_precision_empty_1d() -> None:
    assert objects_are_allclose(
        multilabel_precision(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_precision_empty_2d() -> None:
    assert objects_are_allclose(
        multilabel_precision(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3))),
        {
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_precision_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


def test_multilabel_precision_nan_omit() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="omit",
        ),
        {
            "count": 3,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_omit_y_true() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_omit_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_nan_propagate() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        ),
        {
            "count": 5,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_precision_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 5,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_precision_nan_propagate_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        ),
        {
            "count": 5,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "precision": np.array([]),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_precision_nan_raise() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        multilabel_precision(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="raise",
        )


def test_multilabel_precision_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match=r"'y_true' contains at least one NaN value"):
        multilabel_precision(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            nan_policy="raise",
        )


def test_multilabel_precision_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match=r"'y_pred' contains at least one NaN value"):
        multilabel_precision(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="raise",
        )


#####################################
#     Tests for find_label_type     #
#####################################


def test_find_label_type_binary() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        )
        == "binary"
    )


def test_find_label_type_binary_nans() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        )
        == "binary"
    )


def test_find_label_type_binary_y_true_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        )
        == "binary"
    )


def test_find_label_type_binary_y_pred_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        )
        == "binary"
    )


def test_find_label_type_multiclass() -> None:
    assert (
        find_label_type(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2]))
        == "multiclass"
    )


def test_find_label_type_multiclass_nans() -> None:
    assert (
        find_label_type(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        )
        == "multiclass"
    )


def test_find_label_type_multiclass_y_true_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 1]),
        )
        == "multiclass"
    )


def test_find_label_type_multiclass_y_pred_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([0, 0, 1, 1, 2, 2, 1]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        )
        == "multiclass"
    )


def test_find_label_type_multilabel() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
        == "multilabel"
    )


def test_find_label_type_multilabel_nans() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, float("nan")]]),
        )
        == "multilabel"
    )


def test_find_label_type_multilabel_y_true_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]]),
        )
        == "multilabel"
    )


def test_find_label_type_multilabel_y_pred_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, float("nan"), 1]]),
        )
        == "multilabel"
    )
