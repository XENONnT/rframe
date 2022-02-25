from ..utils import singledispatch
from .base import DatasourceInterface
from .mongo import MongoInterface
from .pandas import PandasInterface


def get_interface(source) -> DatasourceInterface:
    return DatasourceInterface.from_source(source)
