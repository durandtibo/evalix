from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from evalix.metric import (
    binary_fbeta_score,
    fbeta_score,
    multiclass_fbeta_score,
    multilabel_fbeta_score,
)

#################################
#     Tests for fbeta_score     #
#################################


def test_fbeta_score_auto_binary() -> None:
    assert objects_are_equal(
        fbeta_score(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_score_binary() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_score_binary_betas() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            betas=[0.5, 1, 2],
        ),
        {"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0},
    )


def test_fbeta_score_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_f1_suffix": 1.0},
    )


def test_fbeta_score_binary_nan_omit() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            nan_policy="omit",
        ),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_score_binary_nan_propagate() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            nan_policy="propagate",
        ),
        {"count": 6, "f1": float("nan")},
        equal_nan=True,
    )


def test_fbeta_score_binary_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            nan_policy="raise",
        )


def test_fbeta_score_auto_multiclass() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "count": 6,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_score_multiclass() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "count": 6,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_score_multiclass_betas() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
            betas=[0.5, 1, 2],
        ),
        {
            "count": 6,
            "f0.5": np.array([1.0, 1.0, 1.0]),
            "macro_f0.5": 1.0,
            "micro_f0.5": 1.0,
            "weighted_f0.5": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
            "f2": np.array([1.0, 1.0, 1.0]),
            "macro_f2": 1.0,
            "micro_f2": 1.0,
            "weighted_f2": 1.0,
        },
    )


def test_fbeta_score_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_fbeta_score_multiclass_nan_omit() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_score_multiclass_nan_propagate() -> None:
    assert objects_are_equal(
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            nan_policy="propagate",
        ),
        {
            "count": 7,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_fbeta_score_multiclass_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            nan_policy="raise",
        )


def test_fbeta_score_auto_multilabel() -> None:
    assert objects_are_allclose(
        fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_score_multilabel() -> None:
    assert objects_are_allclose(
        fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_score_multilabel_betas() -> None:
    assert objects_are_allclose(
        fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            betas=[0.5, 1, 2],
        ),
        {
            "count": 5,
            "f0.5": np.array([1.0, 1.0, 1.0]),
            "macro_f0.5": 1.0,
            "micro_f0.5": 1.0,
            "weighted_f0.5": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
            "f2": np.array([1.0, 1.0, 1.0]),
            "macro_f2": 1.0,
            "micro_f2": 1.0,
            "weighted_f2": 1.0,
        },
    )


def test_fbeta_score_multilabel_prefix_suffix() -> None:
    assert objects_are_allclose(
        fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_fbeta_score_multilabel_nan_omit() -> None:
    assert objects_are_equal(
        fbeta_score(
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
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_score_multilabel_nan_propagate() -> None:
    assert objects_are_equal(
        fbeta_score(
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
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_fbeta_score_multilabel_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        fbeta_score(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            label_type="multilabel",
            nan_policy="raise",
        )


def test_fbeta_score_label_type_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'label_type': incorrect"):
        fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


########################################
#     Tests for binary_fbeta_score     #
########################################


def test_binary_fbeta_score_correct_1d() -> None:
    assert objects_are_equal(
        binary_fbeta_score(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "f1": 1.0},
    )


def test_binary_fbeta_score_correct_2d() -> None:
    assert objects_are_equal(
        binary_fbeta_score(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "f1": 1.0},
    )


def test_binary_fbeta_score_incorrect() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 0, 0]),
            betas=[0.5, 1, 2],
        ),
        {
            "count": 6,
            "f0.5": 0.4166666666666667,
            "f1": 0.3333333333333333,
            "f2": 0.2777777777777778,
        },
    )


def test_binary_fbeta_score_betas() -> None:
    assert objects_are_equal(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            betas=[0.5, 1, 2],
        ),
        {"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0},
    )


def test_binary_fbeta_score_empty() -> None:
    assert objects_are_equal(
        binary_fbeta_score(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "f1": float("nan")},
        equal_nan=True,
    )


def test_binary_fbeta_score_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_f1_suffix": 1.0},
    )


def test_binary_fbeta_score_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


def test_binary_fbeta_score_nan_omit() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="omit",
        ),
        {"count": 4, "f1": 1.0},
    )


def test_binary_fbeta_score_omit_y_true() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            nan_policy="omit",
        ),
        {"count": 5, "f1": 1.0},
    )


def test_binary_fbeta_score_omit_y_pred() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="omit",
        ),
        {"count": 5, "f1": 1.0},
    )


def test_binary_fbeta_score_nan_propagate() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 6, "f1": float("nan")},
        equal_nan=True,
    )


def test_binary_fbeta_score_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        ),
        {"count": 6, "f1": float("nan")},
        equal_nan=True,
    )


def test_binary_fbeta_score_nan_propagate_y_pred() -> None:
    assert objects_are_allclose(
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 6, "f1": float("nan")},
        equal_nan=True,
    )


def test_binary_fbeta_score_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="raise",
        )


def test_binary_fbeta_score_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            nan_policy="raise",
        )


def test_binary_fbeta_score_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        binary_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            nan_policy="raise",
        )


############################################
#     Tests for multiclass_fbeta_score     #
############################################


def test_multiclass_fbeta_score_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "count": 6,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_fbeta_score(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "count": 6,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
            betas=[0.5, 1, 2],
        ),
        {
            "count": 6,
            "f0.5": np.array([1.0, 0.5555555555555556, 0.0]),
            "macro_f0.5": 0.5185185185185185,
            "micro_f0.5": 0.6666666666666666,
            "weighted_f0.5": 0.5185185185185185,
            "f1": np.array([1.0, 0.6666666666666666, 0.0]),
            "macro_f1": 0.5555555555555555,
            "micro_f1": 0.6666666666666666,
            "weighted_f1": 0.5555555555555555,
            "f2": np.array([1.0, 0.8333333333333334, 0.0]),
            "macro_f2": 0.6111111111111112,
            "micro_f2": 0.6666666666666666,
            "weighted_f2": 0.6111111111111112,
        },
    )


def test_multiclass_fbeta_score_betas() -> None:
    assert objects_are_equal(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            betas=[0.5, 1, 2],
        ),
        {
            "count": 6,
            "f0.5": np.array([1.0, 1.0, 1.0]),
            "macro_f0.5": 1.0,
            "micro_f0.5": 1.0,
            "weighted_f0.5": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
            "f2": np.array([1.0, 1.0, 1.0]),
            "macro_f2": 1.0,
            "micro_f2": 1.0,
            "weighted_f2": 1.0,
        },
    )


def test_multiclass_fbeta_score_empty() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "f1": np.array([]),
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_fbeta_score_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_multiclass_fbeta_score_nan_omit() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="omit",
        ),
        {
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_omit_y_true() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_omit_y_pred() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="omit",
        ),
        {
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_nan_propagate() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "count": 7,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_fbeta_score_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
        ),
        {
            "count": 7,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_fbeta_score_nan_propagate_y_pred() -> None:
    assert objects_are_allclose(
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "count": 7,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_fbeta_score_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="raise",
        )


def test_multiclass_fbeta_score_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            nan_policy="raise",
        )


def test_multiclass_fbeta_score_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        multiclass_fbeta_score(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            nan_policy="raise",
        )


############################################
#     Tests for multilabel_fbeta_score     #
############################################


def test_multilabel_fbeta_score_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_fbeta_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "count": 5,
            "f1": np.array([1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_fbeta_score(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "count": 5,
            "f1": np.array([1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "count": 5,
            "f1": np.array([1.0, 1.0, 0.0]),
            "macro_f1": 0.6666666666666666,
            "micro_f1": 0.6666666666666666,
            "weighted_f1": 0.625,
        },
    )


def test_multilabel_fbeta_score_betas() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            betas=[0.5, 1, 2],
        ),
        {
            "count": 5,
            "f0.5": np.array([1.0, 1.0, 1.0]),
            "macro_f0.5": 1.0,
            "micro_f0.5": 1.0,
            "weighted_f0.5": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
            "f2": np.array([1.0, 1.0, 1.0]),
            "macro_f2": 1.0,
            "micro_f2": 1.0,
            "weighted_f2": 1.0,
        },
    )


def test_multilabel_fbeta_score_empty_1d() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "f1": np.array([]),
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_empty_2d() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3))),
        {
            "count": 0,
            "f1": np.array([]),
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_multilabel_fbeta_score_nan_omit() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="omit",
        ),
        {
            "count": 3,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_omit_y_true() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_omit_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="omit",
        ),
        {
            "count": 4,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_nan_propagate() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        ),
        {
            "count": 5,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_nan_propagate_y_true() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 5,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_nan_propagate_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        ),
        {
            "count": 5,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="raise",
        )


def test_multilabel_fbeta_score_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            nan_policy="raise",
        )


def test_multilabel_fbeta_score_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        multilabel_fbeta_score(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            nan_policy="raise",
        )
