from __future__ import annotations
import inspect
from numpy import isin

import pandas as pd
from pydantic import BaseModel
from pydantic.fields import FieldInfo, ModelField
from typing import Dict, List, Mapping, Optional, Union

from .indexes import BaseIndex, Index, MultiIndex
from .interfaces import get_interface


class InsertionError(Exception):
    pass

class UpdateError(Exception):
    pass


class BaseSchema(BaseModel):

    @classmethod
    def default_datasource(cls):
        return cls.empty_dframe()

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
        ''' Fetches index instance for the given field
        If field_info is index type, returns it
        otherwise return a simple Index instance.
        This allows for queries on non-index fields.
        '''
        if isinstance(name, list):
            return MultiIndex(*[cls.index_for(n) for n in name])

        if name not in cls.__fields__:
            raise KeyError(f'{name} is not a valid' 
                           f'field for schema {cls.__name__}')

        field_info = cls.field_info().get(name, None)
        if not isinstance(field_info, BaseIndex):
            field_info = Index()
        field_info.__set_name__(cls, name)
        return field_info

    @classmethod
    def rframe(cls, datasource=None):
        ''' Contruct a RemoteFrame from this schema and
        datasource.
        '''
        import rframe

        if datasource is None:
            datasource = cls.default_datasource()
        return rframe.RemoteFrame(cls, datasource)

    @classmethod
    def extract_labels(cls, **kwargs):
        ''' Extract query labels from kwargs

        returns extracted labels and remaining kwargs
        '''
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
        ''' Internal find function, performs data validation but
        returns raw dicts, not schema instances.
        '''

        query = cls.compile_query(datasource=datasource, **labels)
        docs = query.execute(limit=_limit, skip=_skip)
        # use schema to validate docs returned by backend
        # FIXME: maybe just pass instances instead of dicts
        docs = [cls(**doc).dict() for doc in docs]
        return docs

    @classmethod
    def find(cls, datasource=None, **labels) -> List["BaseSchema"]:
        ''' Find documents in datasource matching the given labels
        returns List[cls]
        '''
        return [cls(**doc) for doc in  cls._find(datasource, **labels)]

    @classmethod
    def find_df(cls, datasource=None, **labels) -> pd.DateOffset:
        docs = [d.pandas_dict() for d in cls.find(datasource, **labels)]
        df = pd.json_normalize(docs)
        if not len(df):
            df = df.reindex(columns=list(cls.__fields__))
        index_fields = list(cls.get_index_fields())
        if len(index_fields) == 1:
            index_fields = index_fields[0]
        return df.set_index(index_fields)

    @classmethod
    def find_first(cls, datasource=None, n=1, **labels) -> "BaseSchema":
        docs = cls._find(datasource=datasource, _limit=n, **labels)
        return [cls(**doc) for doc in docs]

    @classmethod
    def find_one(cls, datasource=None, **labels) -> "BaseSchema":
        docs = cls.find_first(datasource=datasource, n=1, **labels)
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
            data[name] = index.from_pandas(label)
        return cls(**data)

    @classmethod
    def ensure_index(cls, datasource, **kwargs):
        interface = get_interface(datasource, **kwargs)
        names = list(cls.get_index_fields())
        return interface.ensure_index(datasource, names)

    @classmethod
    def unique(cls, datasource=None, fields: Union[str,List[str]] = None, **labels):
        if fields is None:
            fields = list(cls.get_column_fields())
        elif isinstance(fields, str):
            fields = [fields]
        query = cls.compile_query(datasource, **labels)
        return query.unique(fields)

    @classmethod
    def min(cls, datasource=None, fields: Union[str,List[str]] = None, **labels):
        if fields is None:
            fields = list(cls.get_column_fields())
        elif isinstance(fields, str):
            fields = [fields]
        query = cls.compile_query(datasource, **labels)
        return query.min(fields)

    @classmethod
    def max(cls, datasource=None, fields: Union[str,List[str]] = None, **labels):
        if fields is None:
            fields = list(cls.get_column_fields())
        elif isinstance(fields, str):
            fields = [fields]
        query = cls.compile_query(datasource, **labels)
        return query.max(fields)
        
    @classmethod
    def count(cls, datasource=None, **labels):
        query = cls.compile_query(datasource, **labels)
        return int(query.count())

    @property
    def index_labels(self):
        data = self.dict()
        return {k: data[k] for k in self.get_index_fields()}

    @property
    def index_labels_tuple(self):
        return tuple(v for v in self.index_labels.values())

    @property
    def column_values(self):
        values = self.dict()
        return {attr: values[attr]
                for attr in self.get_column_fields()}

    def save(self, datasource=None, **kwargs):
        if datasource is None:
            datasource = self.default_datasource()
        interface = get_interface(datasource, **kwargs)
        existing = self.find(datasource, **self.index_labels)
        if not existing:
            self.__pre_insert(datasource)
            return interface.insert(self)
        elif len(existing) == 1:
            existing[0].__pre_update(datasource, self)
            return interface.update(self)
        else:
            raise InsertionError('Multiple documents match document '
                                 f'index ({self.index_labels}). '
                                 'Multiple update is not supported.')
            
        

    def __pre_insert(self, datasource):
        '''This method is called  pre insertion 
        if self.save(datasource) was called and a query on datasource
        with self.index_labels did not return any documents.

        raises an InsertionError if user defined checks fail.
        '''
        try:
            self.pre_insert(datasource)
        except Exception as e:
            raise InsertionError(f'Cannot insert new document ({self}).'
                                 f'The schema raised the following exception: {e}')


    def __pre_update(self, datasource, new):
        '''This method is called if new.save(datasource)
        was called and a query on datasource
        with new.index_labels returned this document.

        raises an UpdateError if user defined checks fail.
        '''
        if not self.same_index(new):
            raise UpdateError(f'Cannot update document ({self}) with {new}. '
                              f'The index labels do not match.')
        try:
            self.pre_update(datasource, new=new)
        except Exception as e:
            raise UpdateError(f"Cannot update existing instance ({self}) "
                              f"with new instance ({new}), the schema "
                              f"raised the following exception: {e}")

    def pre_insert(self, datasource):
        '''User defined checks to perform
        prior to document insertion.
        Should raise an exception if insertion
        is disallowed.
        '''
        pass
    
    def pre_update(self, datasource, new):
        '''User defined checks to perform
        prior to document update.
        Should raise an exception if update
        is disallowed.
        '''
        pass

    def same_values(self, other):
        if other is None:
            return False
        if not isinstance(other, BaseSchema):
            return False
        return self.column_values == other.column_values

    def same_index(self, other):
        if other is None:
            return False
        if not isinstance(other, BaseSchema):
            return False
        return self.index_labels == other.index_labels

    def pandas_dict(self):
        data = self.dict()
        for name, label in self.index_labels.items():
            index = self.index_for(name)
            data[name] = index.to_pandas(label)
        return data

    @classmethod
    def empty_dframe(cls):
        columns = list(cls.__fields__)
        indexes = list(cls.get_index_fields())
        if len(indexes) == 1:
            indexes = indexes[0]
        return pd.DataFrame().reindex(columns=columns).set_index(indexes)


    def to_pandas(self):
        index_fields = list(self.get_index_fields())
        if len(index_fields) == 1:
            index_fields = index_fields[0]
        df = pd.DataFrame([self.pandas_dict()])
        return df.set_index(index_fields)
