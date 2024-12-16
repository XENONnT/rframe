# type: ignore
"""
Due to a variety of reasons, sometimes the '==' symbol does not
produce the desired results even when 2 qauntities should be considered
equal, for example due to the  imprecise nature of float point arithmetic
we could have 2 values that are the same but are treated as different,
so we can define an 'are_equal' functions to check if the two values
are 'close enough' to be considered equal.
"""


import math
from numbers import Number

from collections.abc import Iterable, Mapping
from typing import overload
from plum import Dispatcher

from .types import Interval

dispatch = Dispatcher()


@dispatch
def are_equal(x: str, y: str):
    return x == y


@dispatch
def are_equal(x: Number, y: Number):
    if math.isnan(x) and math.isnan(y):
        return True
    return math.isclose(x, y)


@dispatch
def are_equal(x: Iterable, y: Iterable):
    if len(x) != len(y):
        return False
    return all([are_equal(xi, yi) for xi, yi in zip(x, y)])


@dispatch
def are_equal(x: Mapping, y: Mapping):
    if len(x) != len(y):
        return False
    for k, v in x.items():
        if k not in y or not are_equal(v, y[k]):
            return False
    return True


@dispatch
def are_equal(x: Interval, y: Interval):
    return are_equal(x.left, y.left) and are_equal(x.right, y.right)


@dispatch
def are_equal(x: object, y: object):
    if x is None and y is None:
        return True
    return x == y
