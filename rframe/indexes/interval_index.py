from typing import Iterable, Mapping

import pandas as pd
from pydantic import ValidationError

from .base import BaseIndex
from ..types import Interval, TimeInterval


class IntervalIndex(BaseIndex):

    __slots__ = BaseIndex.__slots__ + ("closed",)

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)

        if not issubclass(self.field.type_, Interval):
            raise ValidationError(
                f"{name} field is a interval index but is not of type {type(Interval)}"
            )

    def __init__(self, closed="right", **kwargs):
        super().__init__(**kwargs)
        self.closed = closed

    def label_options(self, query):
        """Collect all valid intervals"""
        labels = query.unique(self.name)
        if not labels:
            return []
        if isinstance(labels[0], dict):
            labels = [self.field.type_(**label) for label in labels]
        labels = sorted(labels, key=lambda x: x.left)

        ivs = labels[:1]
        for iv in labels[1:]:
            # touching intervals can be combined
            if iv.left == ivs[-1].right:
                ivs[-1] = Interval[ivs[-1].left, iv.right]
            else:
                ivs.append(iv)
        return ivs
