


import re
from typing import Dict, List
from pydantic import BaseModel
from pydantic.fields import ModelField, FieldInfo

from .indexes import BaseIndex, Index, MultiIndex
from .indexers import get_indexer

def camel_to_snake(name):
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class InsertionError(Exception):
    pass


class BaseSchema(BaseModel):
    _name: str = ''
    
    def __init_subclass__(cls) -> None:
        if 'name' not in cls.__dict__:
            cls.name = camel_to_snake(cls.__name__)
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

        cls.index = index

    @classmethod
    def field_info(cls) -> Dict[str,FieldInfo]:
        return {name: field.field_info 
                for name, field in cls.__fields__.items()}

    @classmethod
    def get_index_fields(cls) -> Dict[str,ModelField]:
        fields = {}
        for name, field in cls.__fields__.items():
            if isinstance(field.field_info, BaseIndex):
                fields[name] = field
        return fields

    @classmethod
    def get_column_fields(cls) -> Dict[str,ModelField]:
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

    @property
    def index_labels(self):
        data = self.dict()
        return {k: data[k] for k in self.get_index_fields()}

    @classmethod
    def compile_query(cls, datastore, **labels)-> List['BaseSchema']:
        indexes = [cls.index_for(name) for name in labels]
        if len(indexes) == 1:
            index = indexes[0]
            label = labels[index.name]
        else:
            index = MultiIndex(*indexes)
            label = labels

        label = index.validate_label(label)
        indexer = get_indexer(datastore)
        
        return indexer.compile_query(index, label)

    @classmethod
    def find_raw(cls, datastore, **labels)-> List['BaseSchema']:
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
        indexer = get_indexer(datastore)
        
        query =  indexer.compile_query(index, label)
        docs = query.apply(datastore)
        docs = index.reduce(docs, labels)
        return docs

    @classmethod
    def find(cls, datastore, **labels)-> List['BaseSchema']:
        docs = cls.find_raw(datastore, **labels)
        if not docs:
            return []
        
        docs = [cls(**doc) for doc in docs]

        return docs

    @classmethod
    def ensure_index(cls, datastore):
        indexer = get_indexer(datastore)
        names = list(cls.get_index_fields())
        indexer.ensure_index(datastore, names)

    def save(self, datastore):
        indexer = get_indexer(datastore)
        existing = self.find(datastore, **self.index_labels)
        if existing:
            existing[0].pre_update(datastore, self)
        else:
            self.pre_insert(datastore)
        return indexer.insert(datastore, self)

    def pre_insert(self, datastore):
        pass

    def pre_update(self, datastore, new):
        pass
    
    def same_values(self, other):
        return all([getattr(self, attr) == getattr(other, attr)
                     for attr in self.get_column_fields()])

    def pandas_dict(self):
        data = self.dict()
        for name, label in self.index_labels.items():
            index = self.index_for(name)
            data[name] = index.to_pandas(label)
        return data
