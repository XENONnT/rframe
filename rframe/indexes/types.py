import datetime
from typing import ClassVar, Mapping, TypeVar

import pydantic
from pydantic import BaseModel, root_validator, ValidationError

LabelType = TypeVar("LabelType", int, str, datetime.datetime)

# allow up to 8 byte integers
MIN_INTEGER = 0
MAX_INTEGER = int(2**63 -1)
INTEGER_RESOLUTION = 1

# Must fit in 64 bit uint with ns resolution
MIN_DATETIME = datetime.datetime(1677, 9, 22, 0, 0)
MAX_DATETIME = datetime.datetime(2232, 1, 1, 0, 0)
# Must not be truncated by mongodb date type
DATETIME_RESOLUTION = datetime.timedelta(microseconds=1000) 


class Interval(BaseModel):
    _min: ClassVar = None
    _max: ClassVar = None
    _resolution: ClassVar = None

    left: LabelType
    right: LabelType = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_field

    @classmethod
    def _validate_boundary(cls, v):
        if v is None:
            raise TypeError('Interval boundary cannot be None.')

        if v < cls._min:
            raise ValueError(f'{cls} boundary must be larger than {cls._min}.')

        if v > cls._max:
            raise ValueError(f'{cls} boundary must be less than {cls._max}.')


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
            left, right = v, v

        if right is None:
            right = cls._max


        
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
        
        cls._validate_boundary(left)
        
        cls._validate_boundary(right)

        if left > right:
            left, right = right, left

        if (right - left ) < cls._resolution:
            left = left - cls._resolution

        values['left'] = left
        values['right'] = right

        return values

    def overlaps(self, other):
        return self.left < other.right and self.right > other.left

    def __lt__(self, other: 'Interval'):
        if self.right is None:
            return False
        return self.right < other.left

    def __le__(self, other: 'Interval'):
        if self.right is None:
            return False
        return self.right <= other.left

    def __eq__(self, other: 'Interval'):
        return self.overlaps(other)

    def __gt__(self, other: 'Interval'):
        if other.right is None:
            return False
        return self.left > other.right

    def __ge__(self, other: 'Interval'):
        if other.right is None:
            return False
        return self.left >= other.right

class IntegerInterval(Interval):
    _resolution = 1
    _min = MIN_INTEGER
    _max = MAX_INTEGER

    left: int = pydantic.Field(ge=MIN_INTEGER, lt=MAX_INTEGER-_resolution)
    right: int = pydantic.Field(default=MAX_INTEGER, ge=_resolution, lt=MAX_INTEGER)


class TimeInterval(Interval):
    _resolution = DATETIME_RESOLUTION
    _min = MIN_DATETIME
    _max = MAX_DATETIME

    left: datetime.datetime
    right: datetime.datetime = MAX_DATETIME
