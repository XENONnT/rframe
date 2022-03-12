from abc import ABC, abstractmethod

from rframe import interfaces


class BaseDataQuery(ABC):

    @abstractmethod
    def execute(self, limit=None, offset=None):
        pass

    def unique(self, field):
        raise NotImplementedError
    
    def max(self, field):
        raise NotImplementedError
    
    def min(self, field):
        raise NotImplementedError
    

class DatasourceInterface(ABC):
    _INTERFACES = {}

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
        if isinstance(source, str):
            for klass in cls._INTERFACES.values():
                try:
                    interface = klass.from_url(source, *args, **kwargs)
                    break
                except NotImplementedError:
                    pass
            else:
                raise ValueError(f"No interface found for source {source}")
            return interface
  
        for type_ in type(source).mro():
            if type_ in cls._INTERFACES:
                return cls._INTERFACES[type_](source, *args, **kwargs)

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

    def insert_many(self, docs):
        return [self.insert(doc) for doc in docs]
