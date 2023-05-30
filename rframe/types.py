import datetime
import pytz
from typing import ClassVar, Literal, Mapping, Optional, TypeVar, Union

import pydantic
from pydantic import BaseModel, root_validator, ValidationError
import pandas as pd

LabelType = Union[int, str, datetime.datetime]

# allow up to 8 byte integers
MIN_INTEGER = 0
MAX_INTEGER = int(2**63 - 1)
MIN_INTEGER_DELTA = 1

# Must fit in 64 bit uint with ns resolution
MIN_DATETIME = datetime.datetime(1677, 9, 22, 0, 0)
MAX_DATETIME = datetime.datetime(2232, 1, 1, 0, 0)

# Will be truncated by mongodb date type
MIN_TIMEDELTA = datetime.timedelta(microseconds=1000)
MAX_TIMEDELTA = datetime.timedelta(days=106751)


class Interval(BaseModel):
    class Config:
        validate_assignment = True
        frozen = True

    _min: ClassVar = None
    _max: ClassVar = None
    _resolution: ClassVar = None

    left: LabelType
    right: Optional[LabelType] = None
    # closed: Literal['left','right','both'] = 'right'

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_field

    @classmethod
    def _validate_boundary(cls, v):
        if v is None:
            raise TypeError("Interval boundary cannot be None.")

        if v < cls._min:
            raise ValueError(f"{cls} boundary must be larger than {cls._min}.")

        if v > cls._max:
            raise ValueError(f"{cls} boundary must be less than {cls._max}.")

        return v

    @classmethod
    def validate_field(cls, v, field):

        if isinstance(v, cls):
            return v

        if isinstance(v, list) and len(v)==1:
            v = v[0]
        
        if isinstance(v, tuple):
            left, right = v
        elif isinstance(v, Mapping):
            left = v.get("left", None)
            right = v.get("right", None)

        elif hasattr(v, "left") and hasattr(v, "left"):
            left = v.left
            right = v.right
        else:
            left, right = v, v

        if right is None:
            right = cls._max

        return cls(left=left, right=right)

    def __class_getitem__(cls, type_):
        if isinstance(type_, tuple):
            left, right = type_
            if isinstance(left, int):
                return IntegerInterval(left=left, right=right)
            else:
                return TimeInterval(left=left, right=right)

        if not isinstance(type_, type):
            type_ = type(type_)

        if issubclass(type_, int):
            return IntegerInterval

        if issubclass(type_, datetime.datetime):
            return TimeInterval

        raise TypeError(type_)

    @root_validator
    def check_non_zero_length(cls, values):
        left, right = values.get("left"), values.get("right")

        left = cls._validate_boundary(left)

        right = cls._validate_boundary(right)

        if left > right:
            raise ValueError("Interval left must be less than right.")
            # FIXME: maybe  left, right = right, left

        if (right - left) < cls._resolution:
            left = left - cls._resolution

        values["left"] = left
        values["right"] = right

        return values

    def overlaps(self, other):
        return self.left < other.right and self.right > other.left

    def contains(self, label):
        return self.left <= label < self.right
    
    def __lt__(self, other: "Interval"):
        if not isinstance(other, self.__class__):
            raise NotImplementedError

        if self.right is None:
            return False
        return self.right < other.left

    def __le__(self, other: "Interval"):
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        if self.right is None:
            return False
        return self.right <= other.left

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.left == other.left) and (self.right == other.right)

    def __gt__(self, other: "Interval"):
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        if other.right is None:
            return False
        return self.left > other.right

    def __ge__(self, other: "Interval"):
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        if other.right is None:
            return False
        return self.left >= other.right

    def __len__(self):
        return self.right - self.left

    def clone(self, left=None, right=None):
        return self.__class__(left=left or self.left, right=right or self.right)

    def __str__(self):
        return f"{str(self.left)} to {str(self.right)}"


class IntegerInterval(Interval):
    _resolution = 1
    _min = MIN_INTEGER
    _max = MAX_INTEGER

    left: int = pydantic.Field(ge=MIN_INTEGER, lt=MAX_INTEGER - _resolution)
    right: int = pydantic.Field(default=MAX_INTEGER, ge=_resolution, lt=MAX_INTEGER)


class TimeInterval(Interval):
    _resolution = MIN_TIMEDELTA
    _min = MIN_DATETIME
    _max = MAX_DATETIME

    left: datetime.datetime
    right: datetime.datetime = MAX_DATETIME

    @classmethod
    def _validate_boundary(cls, value):
        if isinstance(value, str):
            value = pd.to_datetime(value)

        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if value.tzinfo is not None:
            if value.tzinfo.utcoffset(value) is not None:
                value = value.astimezone(pytz.utc)
            value = value.replace(tzinfo=None)

        if value < cls._min:
            raise ValueError(f"{cls} boundary must be larger than {cls._min}.")

        if value > cls._max:
            raise ValueError(f"{cls} boundary must be less than {cls._max}.")
        
        return value
