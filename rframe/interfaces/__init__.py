from .base import DatasourceInterface
from .mongo import MongoInterface
from .pandas import PandasInterface
from ..utils import singledispatch


def get_interface(source) -> DatasourceInterface:
    return DatasourceInterface.from_source(source)
