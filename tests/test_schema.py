
from rframe import (BaseSchema, Index, InterpolatingIndex,
    Interval, IntervalIndex)
from datetime import datetime

class SimpleSchema(BaseSchema):
    index: int = Index(ge=0,lt=2**8)
    value: float


class InterpolatingSchema(BaseSchema):
    index: float = InterpolatingIndex()
    value: float


class IntegerIntervalSchema(BaseSchema):
    index: Interval[int] = IntervalIndex()
    value: float


class TimeIntervalSchema(BaseSchema):
    index: Interval[datetime] = IntervalIndex()
    value: float



