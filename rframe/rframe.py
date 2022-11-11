"""Main module."""

import pandas as pd

from datetime import datetime
from typing import Any, List, Optional, Tuple, Type, Union
from pandas.core.indexes.frozen import FrozenList

from .indexes import MultiIndex

from .interfaces import get_interface
from .interfaces.pandas import to_pandas
from .schema import BaseSchema, InsertionError, UpdateError
from .utils import camel_to_snake, singledispatchmethod
from .data_accessor import DataAccessor

IndexLabel = Optional[Union[int, float, datetime, str, slice, List]]


class RemoteFrame:
    """Implement basic indexing features similar to a pandas dataframe
    but operates on an arbitrary storage backend
    """

    schema: Type[BaseSchema]
    db: DataAccessor

    _lazy: bool = False
    _index = None

    def __init__(
        self, schema: Type[BaseSchema], datasource: Any = None, lazy=False, **labels
    ) -> None:
        self.schema = schema
        self.db = DataAccessor(schema, datasource)
        self._datasource = datasource
        self._labels = labels
        self._index = None
        self._lazy = lazy

    # @classmethod
    # def from_mongodb(cls, schema, url, db, collection, **kwargs) -> "RemoteFrame":
    #     import pymongo

    #     db = pymongo.MongoClient(url, **kwargs)[db]
    #     collection = db[collection]
    #     return cls(schema, collection)

    @property
    def datasource(self) -> Any:
        if isinstance(self._datasource, RemoteFrame):
            self._datasource = self._datasource.df
        return self._datasource

    @property
    def df(self) -> pd.DataFrame:
        return self.db.find_df(**self._labels)

    @property
    def name(self) -> str:
        return camel_to_snake(self.schema.__name__)

    @property
    def index_names(self) -> FrozenList:
        return FrozenList(self.schema.get_index_fields())

    @property
    def columns(self) -> pd.Index:
        return pd.Index(list(self.schema.get_column_fields()))

    @property
    def index(self) -> pd.Index:
        # FIXME: make a proxy object that only loads labels if accessed.
        if self._index is None:
            self._recreate_index()
        return self._index

    @property
    def loc(self) -> "LocIndexer":
        return LocIndexer(self)

    @property
    def iloc(self) -> "ILocIndexer":
        return ILocIndexer(self)

    @property
    def at(self):
        return AtIndexer(self)

    @property
    def size(self) -> int:
        return self.db.count(**self._labels)

    def head(self, n=10) -> pd.DataFrame:
        """Return first n documents as a pandas dataframe"""
        sort = list(self.schema.get_index_fields())
        return self.db.find_df(limit=n, sort=sort, **self._labels)

    def isel(self, idx: Union[int, slice]):
        """Get a single document or slice by index"""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        if not isinstance(idx, slice):
            raise ValueError("Index must be an integer or slice")
        skip = idx.start if idx.start else 0
        limit = idx.stop - skip if idx.stop is not None else None
        sort = list(self.schema.get_index_fields())
        return self.db.find_df(
            skip=skip, limit=limit, sort=sort, **self._labels
        )

    def unique(self, columns: Union[str, List[str]] = None):
        """Return unique values for each column"""
        if columns is None:
            columns = self.columns
        return self.db.unique(fields=columns, **self._labels)

    def min(self, columns: Union[str, List[str]]) -> Any:
        """Return the minimum value for column"""
        if columns is None:
            columns = self.columns
        return self.db.min(fields=columns, **self._labels)

    def max(self, columns: Union[str, List[str]]) -> Any:
        """Return the maximum value for column"""
        if columns is None:
            columns = self.columns
        return self.db.max(fields=columns, **self._labels)

    def sel(self, *args: IndexLabel, **kwargs: IndexLabel) -> pd.DataFrame:
        """select a subset of the data
        If not lazy or if all indexes are specified, returns a pandas dataframe.
        Otherwise returns a RemoteFrame

        """
        index_fields = self.index_names
        labels = {name: lbl for name, lbl in zip(index_fields, args)}
        labels.update(kwargs)

        merged = dict(self._labels)
        extra = {}
        for k, v in labels.items():
            if k in merged:
                extra[k] = v
            else:
                merged[k] = v

        if self._lazy:
            rf = RemoteFrame(self.schema, self.datasource, lazy=self._lazy, **merged)
            if all([label in merged for label in self.index_names]):
                rf = rf.df
        else:
            rf = self.db.find_df(**merged)

        if len(extra):
            rf = RemoteFrame(self.schema, rf, lazy=self._lazy, **extra)

        return rf

    def set(self, *args: IndexLabel, **kwargs: IndexLabel) -> BaseSchema:
        """Insert data by index"""
        labels = {name: lbl for name, lbl in zip(self.index_names, args)}
        kwargs.update(labels)
        doc = self.schema(**kwargs)
        res = doc.save(self.datasource)
        self._index = None
        return res

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
            if not isinstance(record, self.schema):
                record = self.schema(**record)
            try:
                record.save(self.datasource)
                succeeded.append(record)
            except (InsertionError, UpdateError) as e:
                failed.append(record)
                errors.append(str(e))
        if len(succeeded):
            self._index = None
        return succeeded, failed, errors

    def _recreate_index(self) -> None:
        index = self.schema.get_index()
        query = self.schema.compile_query(self.datasource, **self._labels)
        label_options = to_pandas(index.label_options(query))

        if isinstance(index, MultiIndex):
            self._index = pd.MultiIndex.from_product(label_options, names=index.names)
        else:
            self._index = pd.Index(label_options, name=index.name)

    def __getitem__(self, index: Tuple[IndexLabel, ...]) -> "RemoteSeries":
        if isinstance(index, str) and index in self.columns:
            return RemoteSeries(self, index)
        if isinstance(index, tuple) and index[0] in self.columns:
            return RemoteSeries(self, index[0])[index[1:]]
        raise KeyError(f"{index} is not a dataframe column.")

    def __call__(self, column: str, **index: IndexLabel) -> pd.DataFrame:
        index = tuple(index.get(k, None) for k in self.index_names)
        return self.at[index, column]

    def __dir__(self) -> List[str]:
        return self.columns + super().__dir__()

    def __getattr__(self, name: str) -> "RemoteSeries":
        if name != "columns" and name in self.columns:
            return self[name]
        return super().__getattribute__(name)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"index={self.index_names},"
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
    def loc(self) -> "SeriesLocIndexer":
        return SeriesLocIndexer(self)

    @property
    def iloc(self) -> "SeriesILocIndexer":
        return SeriesILocIndexer(self)

    @property
    def at(self):
        return SeriesAtLocator(self)

    @property
    def index(self):
        return self.frame.index

    @property
    def index_names(self):
        return self.frame.index_names

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

    def __repr__(self) -> str:  # pragma: no cover
        return f"RemoteSeries(index={self.frame.index.names}," f"column={self.column})"

    def __len__(self):
        return len(self.frame)


class Indexer:
    def __init__(self, obj: RemoteFrame):
        self.obj = obj


class ILocIndexer(Indexer):
    def __getitem__(self, index: Union[int, slice]) -> pd.DataFrame:
        return self.obj.isel(index)


class LocIndexer(Indexer):
    def __call__(self, *args: IndexLabel, **kwargs: IndexLabel) -> pd.DataFrame:
        return self.obj.sel(*args, **kwargs)

    def __getitem__(self, index: Tuple[IndexLabel]) -> pd.DataFrame:
        columns = None

        if isinstance(index, tuple) and len(index) == 2:
            index, columns = index[0], index[1]
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

        if len(index) < len(self.obj.index_names):
            KeyError(f"{index} is an under defined index.")

        return self.obj.sel(*index)[column].iloc[0]


class SeriesIndexer:
    def __init__(self, obj: RemoteSeries):
        self.obj = obj


class SeriesLocIndexer(SeriesIndexer):
    def __getitem__(self, index: Union[Tuple[IndexLabel], IndexLabel]) -> pd.DataFrame:
        if not isinstance(index, tuple):
            index = (index,)
        return self.obj.sel(*index)

    def __setitem__(self, index: Union[Tuple[IndexLabel], IndexLabel], value: Any):
        if not isinstance(index, tuple):
            index = (index,)
        self.obj.set(*index, value)

    def __repr__(self) -> str:
        return f"SeriesLocIndexer(index={self.obj.index_names})"


class SeriesILocIndexer(SeriesIndexer):
    def __getitem__(self, index: Union[int, slice]) -> pd.DataFrame:
        return self.obj.iloc[index][self.obj.column]


class SeriesAtLocator(SeriesIndexer):
    def __getitem__(self, index: Union[IndexLabel, Tuple[IndexLabel, ...]]) -> Any:
        if not isinstance(index, tuple):
            index = (index,)

        if any([isinstance(idx, (slice, list, type(None))) for idx in index]):
            raise KeyError(f"{index} is not a unique index.")

        if len(index) < len(self.obj.index_names):
            KeyError(f"{index} is an under defined index.")
        return self.obj.frame.at[index, self.obj.column]

    def __repr__(self) -> str:  # pragma: no cover
        return f"SeriesAtLocator(index={self.obj.index_names})"
