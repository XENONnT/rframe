from abc import ABC, abstractmethod
from ..utils import singledispatchmethod


class BaseDataQuery(ABC):

    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass


class DatasourceIndexer(ABC):
    INTERFACES = {}

    def __init__(self, source) -> None:
        self.source = source

    @classmethod
    def register_indexer(cls, source_type, indexer = None):
        def wrapper(indexer):
            if isinstance(source_type, tuple):
                for t in source_type:
                    cls.register(t, indexer)
                return indexer

            if source_type in cls.INTERFACES:
                raise ValueError(f'Interface for source {source_type} already registered.')
            cls.INTERFACES[source_type] = indexer
            return indexer
        return wrapper(indexer) if indexer is not None else wrapper

    @classmethod
    def from_source(cls, source, *args, **kwargs):
        type_ = type(source)
        if type_ in cls.INTERFACES:
            return cls.INTERFACES[type_](source, *args, **kwargs)
        raise NotImplementedError(f'No implementation for data source of type {type(source)}')

    @abstractmethod
    def compile_query(self, index, label):
        pass
    
    def insert(self, source, doc):
        raise NotImplementedError

    def insert_many(self, source, docs):
        raise NotImplementedError