
import datetime
import pandas as pd
from typing import Generic, Mapping, TypeVar
from numpy import isin

from pydantic import BaseModel, ValidationError

LabelType = TypeVar('LabelType', int, str, datetime.datetime)


class Interval(BaseModel, Generic[LabelType]):
    left: LabelType
    right: LabelType = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_field

    @classmethod
    def validate_field(cls, v, field):
        
        if isinstance(v, tuple):
            left, right = v
        elif isinstance(v, Mapping):
            left = v.get('left', None)
            right = v.get('right', None)
        elif isinstance(v, (pd.Interval, Interval)):
            left = v.left
            right = v.right
        else:
            left, right = v, v
            
        if field.sub_fields:
            left, error = field.sub_fields[0].validate(left, {}, loc='LabelType')
            if error:
                raise ValidationError([error])
            if right is not None:
                right, error = field.sub_fields[0].validate(right, {}, loc='LabelType')
                if error:
                    raise ValidationError([error])
        return cls(left=left, right=right)
