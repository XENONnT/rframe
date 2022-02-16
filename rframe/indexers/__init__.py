from .base import DatasourceIndexer
from .mongo import MongoIndexer
from ..utils import singledispatch


def get_indexer(source) -> DatasourceIndexer:
    return DatasourceIndexer.from_source(source)
