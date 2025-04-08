from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from evalix.testing.fixtures import objectory_available
from evalix.utils.factory import setup_object
from evalix.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

##################################
#     Tests for setup_object     #
##################################


@objectory_available
@pytest.mark.parametrize(
    "module", [deque(), {OBJECT_TARGET: "collections.deque", "iterable": [1, 2, 1, 3]}]
)
def test_setup_object(module: Any) -> None:
    assert isinstance(setup_object(module), deque)


def test_setup_object_object() -> None:
    obj = deque([1, 2, 1, 3])
    assert setup_object(obj) is obj
