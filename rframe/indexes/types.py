import datetime
import pandas as pd
from typing import Mapping, TypeVar
from numpy import isin

from pydantic import BaseModel
import pydantic

LabelType = TypeVar("LabelType", int, str, datetime.datetime)


class Interval(BaseModel):
    left: LabelType
    right: LabelType

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_field

    @classmethod
    def validate_field(cls, v, field):

        if isinstance(v, tuple):
            left, right = v
        elif isinstance(v, Mapping):
            left = v.get("left", None)
            right = v.get("right", None)
        elif hasattr(v, 'left') and hasattr(v, 'right'):
            left = v.left
            right = v.right
        else:
            left, right = v, v

        if left > right:
            left, right = right, left
        return cls(left=left, right=right)

    def __class_getitem__(cls, type_):
        if issubclass(type_, int):
            return IntegerInterval
        if issubclass(type_, datetime.datetime):
            return TimeInterval
        raise TypeError(type_)


class IntegerInterval(Interval):
    left = pydantic.conint(ge=0, lt=int(2**32-1))
    right = pydantic.conint(ge=0, lt=int(2**32-1))

class TimeInterval(Interval):
    left: datetime.datetime
    right: datetime.datetime
