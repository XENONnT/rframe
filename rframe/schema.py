from __future__ import annotations
import inspect

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo, ModelField
from typing import Dict, List, Mapping, Optional, Union

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
    def get_query_signature(cls, default=None):
        params = []
        for name in cls.__fields__:
            for type_ in cls.mro():
                if name in getattr(type_, '__annotations__', {}):
                    label_annotation = type_.__annotations__[name]
                    annotation = Optional[Union[label_annotation,List[label_annotation]]]
                    param = inspect.Parameter(name,
                                            inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                                            default=default,
                                            annotation=annotation)
                    params.append(param)
                    break
        return inspect.Signature(params)

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
    def extract_labels(cls, **kwargs):
        labels = {}
        for name in cls.__fields__:
            label = kwargs.pop(name, None)
            if label is not None:
                labels[name] = label           
        return labels, kwargs

    @classmethod
    def compile_query(cls, datasource=None, **labels):
        if datasource is None:
            datasource = cls.default_datasource()

        labels, kwargs = cls.extract_labels(**labels)
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

        kwargs = {k.lstrip('_'):v for k,v in kwargs.items()}
        interface = get_interface(datasource, **kwargs)

        query = interface.compile_query(index, label)
        return query

    @classmethod
    def _find(cls, datasource=None, _skip=None, _limit=None, **labels) -> List["BaseSchema"]:

        query = cls.compile_query(datasource=datasource, **labels)
        docs = query.execute(limit=_limit, skip=_skip)
        # use schema to validate docs returned by backend
        # FIXME: maybe just pass instances instead of dicts
        docs = [cls(**doc).dict() for doc in docs]
        return docs

    @classmethod
    def find(cls, datasource=None, **labels) -> List["BaseSchema"]:
        return [cls(**doc) for doc in  cls._find(datasource, **labels)]

    @classmethod
    def find_df(cls, datasource=None, **labels) -> pd.DateOffset:
        return pd.json_normalize(cls._find(datasource, **labels))

    @classmethod
    def find_one(cls, datasource=None, **labels) -> "BaseSchema":
        docs = cls._find(datasource, _limit=1, **labels)
        if docs:
            return cls(**docs[0])

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
    def ensure_index(cls, datasource, **kwargs):
        interface = get_interface(datasource, **kwargs)
        names = list(cls.get_index_fields())
        return interface.ensure_index(datasource, names)

    @property
    def index_labels(self):
        data = self.dict()
        return {k: data[k] for k in self.get_index_fields()}

    @property
    def index_labels_tuple(self):
        return tuple(v for v in self.index_labels.values())

    def save(self, datasource=None, **kwargs):
        if datasource is None:
            datasource = self.default_datasource()
        interface = get_interface(datasource, **kwargs)
        existing = self.find(datasource, **self.index_labels)
        if existing:
            existing[0].pre_update(datasource, self)
        else:
            self.pre_insert(datasource)
        return interface.insert(self)

    def pre_insert(self, datasource):
        pass

    def pre_update(self, datasource, new):
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
