import datetime
from typing import Callable, Union

import numpy as np

from ..types import Interval
from ..utils import singledispatch
from .base import BaseIndex


def nn_interpolate(x, xs, ys):
    idx = np.argmin(np.abs(x - np.array(xs)))
    return ys[idx]


@singledispatch
def interpolater(x, xs, ys, kind="linear"):
    raise f"Interpolation on type {type(x)} is not supported."


@interpolater.register(float)
@interpolater.register(int)
def interpolate_number(x, xs, ys, kind="linear"):
    if isinstance(ys[0], (float, int)):
        return np.interp(x, xs, ys)
    return nn_interpolate(x, xs, ys)


@interpolater.register(datetime.datetime)
def interpolate_datetime(x, xs, ys, kind="linear"):
    xs = [x.timestamp() for x in xs]
    x = x.timestamp()
    if isinstance(ys[0], (float, int)):
        return interpolate_number(x, xs, ys, kind=kind)
    return nn_interpolate(x, xs, ys)


class InterpolatingIndex(BaseIndex):
    DOCS_PER_LABEL = 2

    kind: str = "linear"
    neighbours: int = 1
    inclusive: bool = False
    extrapolate: Union[bool, Callable] = False

    def __init__(
        self, kind="linear", neighbours=1, inclusive=False, extrapolate=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.kind = kind
        self.neighbours = neighbours
        self.inclusive = inclusive
        self.extrapolate = extrapolate

    def can_extrapolate(self, index):
        if callable(self.extrapolate):
            return self.extrapolate(index)
        return self.extrapolate

    def reduce(self, docs, label):

        if isinstance(label, list):
            return [d for val in label for d in self.reduce(docs, val)]

        label = self.validate_label(label)

        if not docs or label is None:
            return docs

        x = label.timestamp() if isinstance(label, datetime.datetime) else label

        xs = [self.validate_label(d[self.name]) for d in docs]

        # just convert all datetimes to timestamps to avoid complexity
        # FIXME: maybe properly handle timezones instead
        xs = [xi.timestamp() if isinstance(xi, datetime.datetime) else xi for xi in xs]

        if len(docs) == 1:
            new_document = docs[0]
        else:
            new_document = dict(nn_interpolate(x, xs, docs))
        new_document = dict(new_document, **{self.name: label})

        # If we match exactly, we don't need to interpolate
        if x in xs:
            return [new_document]

        if len(xs) > 1 and max(xs) >= x >= min(xs):
            for yname in self.schema.get_column_fields():
                ys = [d[yname] for d in docs if yname in d]
                if len(ys) != len(xs):
                    continue
                new_document[yname] = interpolater(x, xs, ys, kind=self.kind)
            return [new_document]

        if (x > max(xs) and self.can_extrapolate(new_document)) or x == max(xs):
            return [new_document]

        return []

    def label_options(self, query):
        left = query.min(self.name)
        right = query.max(self.name)
        if left is None or right is None:
            return []
        iv_class = Interval[type(left)]
        return [iv_class(left=left, right=right)]
