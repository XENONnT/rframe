from datetime import datetime
import rframe
from rframe import BaseSchema, Index, InterpolatingIndex, Interval, IntervalIndex
from pydantic import Field

class SimpleSchema(BaseSchema):
    index: int = Index(ge=0, lt=2**8)
    value: float = Field(ge=-2**8, lt=2**8)

class SimpleMultiIndexSchema(BaseSchema):
    index1: int = Index(ge=0, lt=2**8)
    index2: str = Index(min_length=1)
  
    value1: float = Field(ge=-2**8, lt=2**8)
    value2: str

class InterpolatingSchema(BaseSchema):
    index: float = InterpolatingIndex(ge=-2**8, lt=2**8)
    value: float = Field(ge=-2**8, lt=2**8)


class IntegerIntervalSchema(BaseSchema):
    index: Interval[int] = IntervalIndex()
    value: float = Field(ge=-2**8, lt=2**8)


class TimeIntervalSchema(BaseSchema):
    index: Interval[datetime] = IntervalIndex()
    value: float


class MultiIndexSchema(BaseSchema):
    index1: int = Index(ge=0, lt=2**8)
    index2: Interval[int] = IntervalIndex()
    index3: Interval[datetime] = IntervalIndex()
    index4: float = InterpolatingIndex(ge=-2**8, lt=2**8)

    value: float = Field(ge=-2**8, lt=2**8)


SCHEMAS = {rframe.utils.camel_to_snake(klass.__name__): klass
             for klass in rframe.BaseSchema.__subclasses__()}