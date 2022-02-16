

import pandas as pd

from typing import Iterable, Mapping
from nbformat import ValidationError

from .types import Interval
from .base import BaseIndex


class IntervalIndex(BaseIndex):
    __slots__ = BaseIndex.__slots__ + ('left_name', 'right_name', 'closed')

    def __set_name__(self, owner, name):
        self.name = name
        self.schema = owner
        if owner.__fields__[name].type_ is not Interval:
            raise ValidationError(f'{name} is a interval index but is not of type Interval')
        sub_fields = owner.__fields__[name].sub_fields
        if not sub_fields:
            raise ValidationError(f'You must specify a label type for the {name} index')
        self.label_type = sub_fields[0].type_

    def __init__(self, closed='right', **kwargs):
        super().__init__(**kwargs)
        
        self.closed = closed

    def to_pandas(self, label):
        if isinstance(label, Mapping):
            left = pd.to_datetime(label['left'])
            right = pd.to_datetime(label['right'])
            
        elif isinstance(label, Interval):
            left = pd.to_datetime(label.left)
            right = pd.to_datetime(label.right)

        elif isinstance(label, Iterable) and len(label)==2:
            left = pd.to_datetime(label[0])
            right = pd.to_datetime(label[1])
        else:
            raise TypeError(f"{self.name} must be a Mapping,Interval"
             "or Iterable of length 2, got {type(label)} instead.")

        label = pd.Interval(left, right, closed=self.closed)
        return label

    def from_pandas(self, label):
        if isinstance(label, pd.Interval):
            label = Interval(left=label.left, right=label.right)
        return label