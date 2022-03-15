"""Main module."""

from datetime import datetime
from typing import Any, List, Tuple, Type, Union

import pandas as pd
from pydantic.typing import NoneType

from .interfaces import get_interface
from .schema import BaseSchema, InsertionError, UpdateError
from .utils import camel_to_snake, singledispatchmethod

IndexLabel = Union[int, float, datetime, str, slice, NoneType, List]


class RemoteFrame:
    """Implement basic indexing features similar to a pandas dataframe
    but operates on an arbitrary storage backend
    """

    schema: Type[BaseSchema]
    db: Any

    def __init__(self, schema: Type[BaseSchema], datasource: Any = None, **labels) -> None:
        self.schema = schema
        self._datasource = datasource
        self._labels = labels

    @classmethod
    def from_mongodb(cls, schema, url, db, collection, **kwargs):
        import pymongo

        db = pymongo.MongoClient(url, **kwargs)[db]
        collection = db[collection]
        return cls(schema, collection)

    @property
    def datasource(self):
        if isinstance(self._datasource, RemoteFrame):
            self._datasource = self._datasource.df
        return self._datasource

    @property
    def df(self):
        return self.schema.find_df(self.datasource, **self._labels)

    @property
    def name(self):
        return camel_to_snake(self.schema.__name__)

    @property
    def columns(self):
        return list(self.schema.get_column_fields())

    @property
    def index(self):
        return self.schema.get_index()

    @property
    def loc(self):
        return LocIndexer(self)

    @property
    def at(self):
        return AtIndexer(self)

    @property
    def size(self):
        return self.schema.count(self.datasource, **self._labels)

    def head(self, n=10) -> pd.DataFrame:
        """Return first n documents as a pandas dataframe"""
        docs = self.schema.find_first(self.datasource, n=n, **self._labels)
        index_fields = list(self.schema.get_index_fields())
        all_fields = index_fields + self.columns
        df = pd.DataFrame([doc.pandas_dict() for doc in docs], columns=all_fields)
        idx = [c for c in index_fields if c in df.columns]
        return df.sort_values(idx).set_index(idx)

    def unique(self, columns: Union[str, List[str]] = None):
        """Return unique values for each column"""
        if columns is None:
            columns = self.columns
        return self.schema.unique(self.datasource, fields=columns, **self._labels)

    def min(self, columns: Union[str, List[str]]) -> Any:
        """Return the minimum value for column"""
        if columns is None:
            columns = self.columns
        return self.schema.min(self.datasource, fields=columns, **self._labels )

    def max(self, columns: Union[str, List[str]]) -> Any:
        """Return the maximum value for column"""
        if columns is None:
            columns = self.columns
        return self.schema.max(self.datasource, fields=columns, **self._labels)

    def sel(self, *args: IndexLabel, **kwargs: IndexLabel) -> pd.DataFrame:
        """select a subset of the data
        returns a pandas dataframe if all indexes are specified.
        Otherwise returns a RemoteFrame
        """
        index_fields = self.index.names
        labels = {name: lbl for name, lbl in zip(index_fields, args)}
        labels.update(kwargs)

        merged = dict(self._labels)
        extra = {}
        for k,v in labels.items():
            if k in merged:
                extra[k] = v
            else:
                merged[k] = v

        rf = RemoteFrame(self.schema, self.datasource, **merged)
        if all([label in merged for label in self.index.names]):
            rf = rf.df
        if len(extra):
            rf = RemoteFrame(self.schema, rf, **extra)

        return rf

    def set(self, *args: IndexLabel, **kwargs: IndexLabel) -> BaseSchema:
        """Insert data by index"""
        labels = {name: lbl for name, lbl in zip(self.index.names, args)}
        kwargs.update(labels)
        doc = self.schema(**kwargs)
        return doc.save(self.datasource)

    def append(
        self, records: Union[pd.DataFrame, List[dict]]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """Insert multiple records into the DB"""
        if isinstance(records, RemoteFrame):
            records = records.df
        if isinstance(records, pd.DataFrame):
            records = self.schema.from_pandas(records)
        
        succeeded = []
        failed = []
        errors = []
        for record in records:
            if not isinstance(doc, self, self.schema):
                doc = self.schema(**record)
            try:
                doc.save(self.datasource)
                succeeded.append(doc)
            except (InsertionError, UpdateError) as e:
                failed.append(doc)
                errors.append(str(e))

        return succeeded, failed, errors

    def __getitem__(self, index: Tuple[IndexLabel, ...]) -> "RemoteSeries":
        if isinstance(index, str) and index in self.columns:
            return RemoteSeries(self, index)
        if isinstance(index, tuple) and index[0] in self.columns:
            return RemoteSeries(self, index[0])[index[1:]]
        raise KeyError(f"{index} is not a dataframe column.")

    def __call__(self, column: str, **index: IndexLabel) -> pd.DataFrame:
        index = tuple(index.get(k, None) for k in self.index.names)
        return self.at[index, column]

    def __dir__(self) -> List[str]:
        return self.columns + super().__dir__()

    def __getattr__(self, name: str) -> "RemoteSeries":
        if name != "columns" and name in self.columns:
            return self[name]
        raise AttributeError(name)

    def __len__(self):
        return self.size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"index={self.index.names},"
            f"columns={self.columns},"
            f"datasource={self._datasource},"
            f"selection={self._labels})"
        )


class RemoteSeries:
    frame: RemoteFrame
    column: str

    def __init__(self, frame: RemoteFrame, column: str) -> None:
        self.frame = frame
        self.column = column

    def __getitem__(
        self, index: Union[IndexLabel, Tuple[IndexLabel, ...]]
    ) -> pd.DataFrame:
        if not isinstance(index, tuple):
            index = (index,)
        return self.frame.sel(*index)[self.column]

    @property
    def loc(self):
        return SeriesLocIndexer(self)

    @property
    def at(self):
        return SeriesAtLocator(self)

    @property
    def index(self):
        return self.frame.index

    def sel(self, *args: IndexLabel, **kwargs: IndexLabel) -> pd.DataFrame:
        rf = self.frame.sel(*args, **kwargs)
        if isinstance(rf, RemoteFrame):
            rf = rf.df
        return rf[self.column]

    def set(self, *args: IndexLabel, **kwargs: IndexLabel):
        raise InsertionError(
            "Cannot set values on a RemoteSeries object," "use the RemoteDataFrame."
        )

    def unique(self):
        return self.frame.unique(self.column)

    def min(self):
        return self.frame.min(self.column)

    def max(self):
        return self.frame.max(self.column)

    def __repr__(self) -> str:
        return f"RemoteSeries(index={self.frame.index.names}," f"column={self.column})"

    def __len__(self):
        return len(self.frame)


class Indexer:
    def __init__(self, obj: RemoteFrame):
        self.obj = obj


class LocIndexer(Indexer):
    def __call__(self, *args: IndexLabel, **kwargs: IndexLabel) -> pd.DataFrame:
        return self.obj.sel(*args, **kwargs)

    def __getitem__(self, index: Tuple[IndexLabel]) -> pd.DataFrame:
        columns = None

        if isinstance(index, tuple) and len(index) == 2:
            index, columns = index
            if not isinstance(columns, list):
                columns = [columns]
            if not all([c in self.obj.columns for c in columns]):
                if not isinstance(index, tuple):
                    index = (index,)
                index = index + tuple(columns)
                columns = None

        elif isinstance(index, tuple) and len(index) == len(self.obj.columns) + 1:
            index, columns = index[:-1], index[-1]

        if not isinstance(index, tuple):
            index = (index,)

        df = self.obj.sel(*index)

        if columns is not None:
            df = df[columns]

        return df

    def __setitem__(self, key: Any, value: Union[dict, BaseSchema]) -> BaseSchema:
        if not isinstance(key, tuple):
            key = (key,)

        if isinstance(value, self.obj.schema):
            value = value.dict()

        if not isinstance(value, dict):
            value = {"value": value}

        return self.obj.set(*key, **value)


class SeriesLocIndexer(Indexer):
    def __init__(self, obj: RemoteSeries):
        super().__init__(obj)

    def __getitem__(self, index: IndexLabel) -> pd.DataFrame:
        return self.obj.sel(*index)

    def __setitem__(self, index: IndexLabel, value: Any):
        self.obj.set(*index, value)

    def __repr__(self) -> str:
        return f"SeriesLocIndexer(index={self.obj.index.names})"


class AtIndexer(Indexer):
    def __getitem__(self, key: Tuple[Tuple[IndexLabel, ...], str]) -> Any:

        if not (isinstance(key, tuple) and len(key) == 2):
            raise KeyError(
                "ill-defined location. Specify "
                ".at[index,column] where index can be a tuple."
            )

        index, column = key

        if column not in self.obj.columns:
            raise KeyError(f"{column} not found. Valid columns are: {self.obj.columns}")

        if not isinstance(index, tuple):
            index = (index,)

        if any([isinstance(idx, (slice, list, type(None))) for idx in index]):
            raise KeyError(f"{index} is not unique index.")

        if len(index) < len(self.obj.index.names):
            KeyError(f"{index} is an under defined index.")

        return self.obj.sel(*index)[column].iloc[0]


class SeriesAtLocator(AtIndexer):
    def __init__(self, obj: RemoteSeries):
        super().__init__(obj)

    def __getitem__(self, index: IndexLabel) -> Any:
        if not isinstance(index, tuple):
            index = (index,)

        if any([isinstance(idx, (slice, list, type(None))) for idx in index]):
            raise KeyError(f"{index} is not a unique index.")

        if len(index) < len(self.obj.index.names):
            KeyError(f"{index} is an under defined index.")
        return self.obj.frame.at[index, self.obj.column]

    def __repr__(self) -> str:
        return f"SeriesAtLocator(index={self.obj.index.names})"