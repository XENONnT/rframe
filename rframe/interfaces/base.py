from abc import ABC, abstractmethod
from typing import Dict, List, Union
from rframe import interfaces
from loguru import logger
from ..utils import filter_kwargs


class BaseDataQuery(ABC):
    @abstractmethod
    def execute(self, limit=None, skip=None, sort=None):  # pragma: no cover
        pass

    def iter(self, limit=None, skip=None, sort=None):
        yield from self.execute(limit=limit, skip=skip, sort=sort)

    def paginate(self, page_size=100, limit=None, skip=None, sort=None):
        docs = []
        for doc in self.iter(limit=limit, skip=skip, sort=sort):
            docs.append(doc)
            if len(docs) == page_size:
                yield docs
                docs = []
        if docs:
            yield docs

    def unique(self, fields: Union[str, List[str]]):
        raise NotImplementedError

    def max(self, fields: Union[str, List[str]]):
        raise NotImplementedError

    def min(self, fields: Union[str, List[str]]):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError


class DatasourceInterface(ABC):
    _INTERFACES: Dict[str, "DatasourceInterface"] = {}

    def __init__(self, source) -> None:
        self.source = source

    @classmethod
    def register_interface(cls, source_type, interface=None):
        def wrapper(interface):
            if isinstance(source_type, tuple):
                for t in source_type:
                    cls.register(t, interface)
                return interface

            if source_type in cls._INTERFACES:
                raise ValueError(
                    f"Interface for source {source_type} already registered."
                )
            cls._INTERFACES[source_type] = interface
            return interface

        return wrapper(interface) if interface is not None else wrapper

    @classmethod
    def from_source(cls, source, *args, **kwargs):
        logger.debug(
            f"Looking for interface for datasource: {source}, " f"with kwargs: {kwargs}"
        )

        if isinstance(source, str):
            for klass in cls._INTERFACES.values():
                try:
                    filtered_kwargs = filter_kwargs(klass.from_url, kwargs)
                    interface = klass.from_url(source, *args, **filtered_kwargs)
                    logger.info(f"Found interface {klass}.")
                    break
                except NotImplementedError:
                    pass
            else:
                raise ValueError(f"No interface found for source {source}")

            return interface

        for type_ in type(source).mro():
            if type_ in cls._INTERFACES:
                interface_class = cls._INTERFACES[type_]
                logger.info(f"Found interface {interface_class}.")
                filtered_kwargs = filter_kwargs(interface_class, kwargs)
                return interface_class(source, *args, **filtered_kwargs)

        raise NotImplementedError(
            f"No implementation for data source of type {type(source)}"
        )

    @classmethod
    def from_url(cls, url: str, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compile_query(self, index, label):
        pass

    def insert(self, doc):
        raise NotImplementedError

    def insert_many(self, docs: list) -> list:
        return [self.insert(doc) for doc in docs]

    def update(self, doc):
        raise NotImplementedError

    def update_many(self, docs: list) -> list:
        return [self.update(doc) for doc in docs]

    def delete(self, doc):
        raise NotImplementedError

    def initdb(self, schema):
        raise NotImplementedError
