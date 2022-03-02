from abc import ABC, abstractmethod


class BaseDataQuery(ABC):
    @abstractmethod
    def apply(self):
        pass


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
        for type_ in type(source).mro():
            if type_ in cls._INTERFACES:
                return cls._INTERFACES[type_](source, *args, **kwargs)

        raise NotImplementedError(
            f"No implementation for data source of type {type(source)}"
        )

    @abstractmethod
    def compile_query(self, index, label):
        pass

    def insert(self, source, doc):
        raise NotImplementedError

    def insert_many(self, source, docs):
        raise NotImplementedError
