
import pytz
import numbers
import pandas as pd
from pydantic import ValidationError
from pydantic.fields import ModelField, FieldInfo
from typing import Generic, Any

from datetime import datetime

from rframe.utils import singledispatchmethod
from .types import LabelType


class BaseIndex(FieldInfo):
    __slots__ = FieldInfo.__slots__ + ('name', 'schema', 'label_type', 'nullable')

    def __init__(self, default: Any = ..., **kwargs: Any) -> None:
        self.nullable = kwargs.pop('nullable', True)
        super().__init__(default, **kwargs)
        self.name = 'index'
        self.schema = None
        self.label_type = Any
        
    def __set_name__(self, owner, name):
        self.name = name
        self.schema = owner
        self.label_type = owner.__fields__[name].type_

    @property
    def names(self):
        return [self.name]

    def validate_label(self, label):
        if isinstance(label, slice):
            start = self.validate_label(label.start)
            stop = self.validate_label(label.stop)
            step = self.validate_label(label.step)
            if start is None and stop is None:
                label = None
            else:
                return slice(start, stop, step)

        if label is None and self.nullable:
            return label

        if isinstance(label, dict) and self.label_type is not dict:
            return {k: self.validate_label(val) for k,val in label.items()}

        if isinstance(label, list) and self.label_type is not list:
            return [self.validate_label(val) for val in label]

        if isinstance(label, tuple) and self.label_type is not tuple:
            return tuple(self.validate_label(val) for val in label)

        label = self.coerce(label)
        
        if not isinstance(label, self.label_type):
            raise TypeError(f'{self.name} must be of type {self.label_type}')

        return label

    def coerce(self, label):
        if isinstance(label, self.label_type):
            return label
        
        label =  self._coerce(self.label_type, label)

        return label

    @singledispatchmethod
    def _coerce(self, type_, value):
        return type_(value)

    @_coerce.register(datetime)
    def _coerce_datetime(self, type_, value):
        if isinstance(value, datetime):
            if value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None:
                value = value.astimezone(pytz.utc)
            else:
                value = value.replace(tzinfo=pytz.utc)
            return value
        unit = self.unit if isinstance(value, numbers.Number) else None
        utc = getattr(self, 'utc', True)
        value = pd.to_datetime(value, utc=utc, unit=unit).to_pydatetime()
        return self._coerce_datetime(type_, value)

    def to_pandas(self, label):
        return label

    def from_pandas(self, label):
        return label

    def reduce(self, docs, labels):
        return docs

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},type={self.label_type})"

