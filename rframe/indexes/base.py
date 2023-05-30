import numbers
from datetime import datetime
from typing import Any

import pandas as pd
import pytz
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo, Undefined
from rframe.types import Interval

from rframe.utils import singledispatchmethod


class BaseIndex(FieldInfo):
    DOCS_PER_LABEL = 1

    __slots__ = FieldInfo.__slots__ + ("name", "schema", "field", "nullable", "unique")

    def __init__(self, default: Any = Undefined, **kwargs: Any) -> None:
        self.nullable = kwargs.pop("nullable", True)
        self.unique = kwargs.pop("unique", True)
        super().__init__(default, **kwargs)
        self.name = "index"
        self.schema = None
        self.field = None

    def __set_name__(self, owner, name):
        self.name = name
        self.schema = owner
        self.field = owner.__fields__[name]

    @property
    def names(self):
        """returns a list of names
        for compatibility with multi-index
        """
        return [self.name]

    def _validate_label(self, label):
        """Perform label validation"""
        if label is None:
            return label

        if isinstance(label, pd.Timestamp):
            label = label.to_pydatetime()
        label, error = self.field.validate(label, {}, loc=self.name)
        if error:
            raise ValidationError([error], self.schema)
        if isinstance(label, BaseModel):
            label = label.dict()
        return label

    def validate_label(self, label):
        if isinstance(label, dict) and self.name in label:
            label = label[self.name]
        
        if isinstance(label, dict) and self.field.type_ is not dict:
            return {k: self._validate_label(v) for k, v in label.items()}
        
        if isinstance(label, Interval) and not issubclass(self.field.type_, Interval):
            left = self._validate_label(label.left)
            right = self._validate_label(label.right)
            return label.__class__(left=left, right=right)
        
        if isinstance(label, slice):
            start = self._validate_label(label.start)
            stop = self._validate_label(label.stop)
            step = self._validate_label(label.step)

            if start is None and stop is None:
                label = None
            elif step is not None:
                return list(range(start, stop, step))
            else:
                return slice(start, stop, step)

        if isinstance(label, list) and self.field.type_ is not list:
            return [self._validate_label(val) for val in label]

        if isinstance(label, tuple) and self.field.type_ is not tuple:
            return tuple(self._validate_label(val) for val in label)

        return self._validate_label(label)

    def reduce(self, docs, labels):
        return docs

    def label_options(self, query):
        return query.unique(self.name)

    def __repr__(self):
        type_ = self.field.type_ if self.field else "UNKNOWN"
        return f"{self.__class__.__name__}(name={self.name},type={type_})"
