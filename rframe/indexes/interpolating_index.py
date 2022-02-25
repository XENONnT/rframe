import datetime
from typing import Callable, Union

import numpy as np
from scipy.interpolate import interp1d

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
        func = interp1d(xs,
                        ys,
                        fill_value=(ys[0], ys[-1]),
                        bounds_error=False,
                        kind=kind)
        return func(x).item()
    return nn_interpolate(x, xs, ys)


@interpolater.register(datetime.datetime)
def interpolate_datetime(x, xs, ys, kind="linear"):
    xs = [x.timestamp() for x in xs]
    x = x.timestamp()
    if isinstance(ys[0], (float, int)):
        return interpolate_number(x, xs, ys, kind=kind)
    return nn_interpolate(x, xs, ys)


class InterpolatingIndex(BaseIndex):
    kind: str = "linear"
    neighbours: int = 1
    inclusive: bool = False
    extrapolate: Union[bool, Callable] = False

    def __init__(self,
                 kind="linear",
                 neighbours=1,
                 inclusive=False,
                 extrapolate=False,
                 **kwargs):
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
        if label is None:
            return docs

        if isinstance(label, list):
            return [d for val in label for d in self.reduce(docs, val)]

        label = self.validate_label(label)
        x = label.timestamp() if isinstance(label,
                                            datetime.datetime) else label

        xs = [self.validate_label(d[self.name]) for d in docs]

        # just convert all datetimes to timestamps to avoid complexity
        # FIXME: maybe properly handle timezones instead
        xs = [
            xi.timestamp() if isinstance(xi, datetime.datetime) else xi
            for xi in xs
        ]

        new_document = dict(nn_interpolate(x, xs, docs))
        new_document[self.name] = label

        if len(xs) > 1 and max(xs) >= x >= min(xs):
            for yname in self.schema.get_column_fields():
                ys = [d[yname] for d in docs]
                new_document[yname] = interpolater(x, xs, ys, kind=self.kind)
            return [new_document]

        if (x > max(xs)
                and self.can_extrapolate(new_document)) or x == max(xs):
            return [new_document]

        return []
