from pydantic import ValidationError
from pydantic.fields import ModelField
import datetime
from typing import TypeVar, Generic

from rframe.utils import singledispatchmethod
from .base import BaseIndex
from .types import LabelType
# IndexType = TypeVar('IndexType', int, str, datetime.datetime)


class Index(BaseIndex):
    pass
    
