from loguru import logger

from ..utils import singledispatch
from .base import DatasourceInterface

from .pandas import PandasInterface
from .rest import RestInterface
from .json import JsonInterface

try:
    from .mongo import MongoInterface
except ImportError:
    pass

try:
    from .tinydb import TinyDBInterface
except ImportError:
    pass


def get_interface(source, **kwargs) -> DatasourceInterface:
    return DatasourceInterface.from_source(source, **kwargs)
