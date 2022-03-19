from loguru import logger

from ..utils import singledispatch
from .base import DatasourceInterface
from .mongo import MongoInterface
from .pandas import PandasInterface
from .rest import RestInterface
from .tinydb import TinyDBInterface
from .json import JsonInterface


def get_interface(source, **kwargs) -> DatasourceInterface:
    return DatasourceInterface.from_source(source, **kwargs)
