import datetime
from typing import ClassVar, Mapping, TypeVar

import pydantic
from pydantic import BaseModel, root_validator

LabelType = TypeVar("LabelType", int, str, datetime.datetime)


class Interval(BaseModel):
    _resolution: ClassVar = None

    left: LabelType
    right: LabelType = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_field

    @classmethod
    def validate_field(cls, v, field):
        if isinstance(v, cls):
            return v

        if isinstance(v, tuple):
            left, right = v
        elif isinstance(v, Mapping):
            left = v.get("left", None)
            right = v.get("right", None)

        elif hasattr(v, "left") and hasattr(v, "left"):
            left = v.left
            right = v.right
        else:
            if cls._resolution is not None:
                left, right = v, v
            else:
                left, right = v, v

        return cls(left=left, right=right)

    def __class_getitem__(cls, type_):
        if issubclass(type_, int):
            return IntegerInterval
        if issubclass(type_, datetime.datetime):
            return TimeInterval
        raise TypeError(type_)
        
    @root_validator
    def check_non_zero_length(cls, values):
        left, right = values.get('left'), values.get('right')

        if right is not None and left > right:
            left, right = right, left

        if (right - left ) < cls._resolution:
            left = left - cls._resolution

        values['left'] = left
        values['right'] = right 
        return values


class IntegerInterval(Interval):
    _resolution = 1

    left = pydantic.conint(ge=0, lt=int(2**32 - 1))
    right = pydantic.conint(ge=0, lt=int(2**32 - 1))


class TimeInterval(Interval):
    _resolution = datetime.timedelta(microseconds=1000)

    left: datetime.datetime
    right: datetime.datetime = None
