import pandas as pd

from typing import Iterable, Mapping
from nbformat import ValidationError

from .types import Interval, TimeInterval
from .base import BaseIndex


class IntervalIndex(BaseIndex):
    __slots__ = BaseIndex.__slots__ + ("closed",)

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

        if not issubclass(self.field.type_, Interval):
            raise ValidationError(
                f"{name} field is a interval index but is not of type {type(Interval)}"
            )
        # if not self.field.sub_fields:
        #     raise ValidationError(f'You must specify a label type for the {name} index')

    def __init__(self, closed="right", **kwargs):
        super().__init__(**kwargs)

        self.closed = closed

    def _to_pandas(self, label):

        if self.field.type_ is TimeInterval:
            if label is None:
                label = 9e18
            return pd.to_datetime(label)
        if label is None:
            label = float("inf")
        return label

    def to_pandas(self, label):
        if isinstance(label, Mapping):
            left = self._to_pandas(label["left"])
            right = self._to_pandas(label["right"])

        elif isinstance(label, Interval):
            left = self._to_pandas(label.left)
            right = self._to_pandas(label.right)

        elif isinstance(label, Iterable) and len(label) == 2:
            left = self._to_pandas(label[0])
            right = self._to_pandas(label[1])
        else:
            raise TypeError(
                f"{self.name} must be a Mapping,Interval"
                "or Iterable of length 2, got {type(label)} instead."
            )

        label = pd.Interval(left, right, closed=self.closed)
        return label

    def from_pandas(self, label):
        if isinstance(label, pd.Interval):
            label = Interval(left=label.left, right=label.right)
        return label
