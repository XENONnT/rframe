from typing import Dict, List, Mapping

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo, ModelField

from .indexes import BaseIndex, Index, MultiIndex
from .interfaces import get_interface


class InsertionError(Exception):
    pass


class BaseSchema(BaseModel):
    @classmethod
    def default_datasource(cls):
        raise NotImplementedError

    @classmethod
    def field_info(cls) -> Dict[str, FieldInfo]:
        return {name: field.field_info for name, field in cls.__fields__.items()}

    @classmethod
    def get_index_fields(cls) -> Dict[str, ModelField]:
        fields = {}
        for name, field in cls.__fields__.items():
            if isinstance(field.field_info, BaseIndex):
                fields[name] = field
        return fields

    @classmethod
    def get_column_fields(cls) -> Dict[str, ModelField]:
        fields = {}
        for name, field in cls.__fields__.items():
            if not isinstance(field.field_info, BaseIndex):
                fields[name] = field
        return fields

    @classmethod
    def get_index(cls) -> BaseIndex:
        index_fields = cls.get_index_fields()
        indexes = []
        for name, field in index_fields.items():
            index = field.field_info
            index.__set_name__(cls, name)
            indexes.append(index)

        if len(indexes) == 1:
            index = indexes[0]
        else:
            index = MultiIndex(*indexes)

        return index

    @classmethod
    def index_for(cls, name):
        field_info = cls.field_info().get(name, None)
        if not isinstance(field_info, BaseIndex):
            field_info = Index()
        field_info.__set_name__(cls, name)
        return field_info

    @classmethod
    def rframe(cls, datasource=None):
        import rframe

        if datasource is None:
            datasource = cls.default_datasource()
        return rframe.RemoteFrame(cls, datasource)

    @classmethod
    def compile_query(cls, datastore, **labels) -> List["BaseSchema"]:
        indexes = [cls.index_for(name) for name in labels]
        if len(indexes) == 1:
            index = indexes[0]
            label = labels[index.name]
        else:
            index = MultiIndex(*indexes)
            label = labels

        label = index.validate_label(label)

        interface = get_interface(datastore)

        return interface.compile_query(index, label)

    @classmethod
    def _find(cls, datastore, **labels) -> List["BaseSchema"]:
        labels = dict(labels)
        for name in cls.get_index_fields():
            if name not in labels:
                labels[name] = None
        indexes = [cls.index_for(name) for name in labels]
        if len(indexes) == 1:
            index = indexes[0]
            label = labels[index.name]
        else:
            index = MultiIndex(*indexes)
            label = labels

        label = index.validate_label(label)
        interface = get_interface(datastore)

        query = interface.compile_query(index, label)
        docs = query.apply(datastore)
        docs = index.reduce(docs, labels)
        return docs

    @classmethod
    def find(cls, datastore=None, **labels) -> List["BaseSchema"]:
        if datastore is None:
            datastore = cls.default_datasource()
        docs = cls._find(datastore, **labels)
        if not docs:
            return []

        docs = [cls(**doc) for doc in docs]

        return docs

    @classmethod
    def find_one(cls, datastore=None, **labels) -> "BaseSchema":
        docs = cls.find(datastore, **labels)
        if docs:
            return docs[0]

    @classmethod
    def from_pandas(cls, record):
        if isinstance(record, list):
            return [cls.from_pandas(d) for d in record]
        if isinstance(record, pd.DataFrame):
            return [cls.from_pandas(d) for d in record.to_dict(orient="records")]

        if not isinstance(record, Mapping):
            raise TypeError(
                "Record must be of type Mapping,"
                "List[Mapping] or DataFrame],"
                f"got {type(record)}"
            )
        data = dict(record)
        for name in cls.get_index_fields():
            index = cls.index_for(name)
            label = record.get(name, None)
            data[name] = index.to_pandas(label)
        return cls(**data)

    @classmethod
    def ensure_index(cls, datastore):
        interface = get_interface(datastore)
        names = list(cls.get_index_fields())
        return interface.ensure_index(datastore, names)

    @property
    def index_labels(self):
        data = self.dict()
        return {k: data[k] for k in self.get_index_fields()}

    @property
    def index_labels_tuple(self):
        return tuple(v for v in self.index_labels.values())

    def save(self, datastore=None):
        if datastore is None:
            datastore = self.default_datasource()
        interface = get_interface(datastore)
        existing = self.find(datastore, **self.index_labels)
        if existing:
            existing[0].pre_update(datastore, self)
        else:
            self.pre_insert(datastore)
        return interface.insert(datastore, self)

    def pre_insert(self, datastore):
        pass

    def pre_update(self, datastore, new):
        pass

    def same_values(self, other):
        for attr in self.get_column_fields():
            left, right = getattr(self, attr), getattr(other, attr)
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                return False
        return True

    def pandas_dict(self):
        data = self.dict()
        for name, label in self.index_labels.items():
            index = self.index_for(name)
            data[name] = index.to_pandas(label)
        return data
