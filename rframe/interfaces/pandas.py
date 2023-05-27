from functools import singledispatch
import numbers
from loguru import logger
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pytz

from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import singledispatchmethod
from .base import BaseDataQuery, DatasourceInterface
from ..types import Interval


class PandasBaseQuery(BaseDataQuery):
    def __init__(self, index, df, column: str, label: Any) -> None:
        self.index = index
        self.df = df
        self.column = column
        self.label = label

    @property
    def labels(self):
        return {self.column: self.label}

    def apply_selection(self, df, label):
        raise NotImplementedError

    def execute(self, limit: int = None, skip: int = None, sort=None):
        logger.debug("Applying pandas dataframe selection")

        if not len(self.df):
            return []
        if sort is None:
            df = self.df
        else:
            if isinstance(sort, str):
                sort = [sort]
            elif isinstance(sort, dict):
                sort = list(sort)
            df = self.df.sort_values(sort)
        df = self.apply_selection(df, self.label)

        if df.index.names or df.index.name:
            df = df.reset_index()
        if limit is not None:
            start = skip * self.index.DOCS_PER_LABEL if skip is not None else 0
            limit = start + limit * self.index.DOCS_PER_LABEL
            df = df.iloc[start:limit]
        docs = df.to_dict(orient="records")
        docs = self.index.reduce(docs, self.labels)
        logger.debug(f"Done. Found {len(docs)} documents.")
        docs = from_pandas(docs)
        return docs

    def min(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        df = self.apply_selection(self.df, self.label)
        results = {}
        for field in fields:
            if field in df.index.names:
                df = df.reset_index()
            results[field] = df[field].min()
        if len(fields) == 1:
            results = results[fields[0]]
        results = from_pandas(results)
        return results

    def max(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        df = self.apply_selection(self.df, self.label)
        results = {}
        for field in fields:
            if field in df.index.names:
                df = df.reset_index()
            results[field] = df[field].max()
        if len(fields) == 1:
            results = results[fields[0]]
        results = from_pandas(results)
        return results

    def unique(
        self, fields: Union[str, List[str]]
    ) -> Union[List[Any], Dict[str, List[Any]]]:
        if isinstance(fields, str):
            fields = [fields]
        df = self.apply_selection(self.df, self.label)

        results = {}
        for field in fields:
            if field in df.index.names:
                df = df.reset_index()
            results[field] = list(df[field].unique())

        if len(fields) == 1:
            results = results[fields[0]]

        results = from_pandas(results)
        return results

    def count(self):
        df = self.apply_selection(self.df, self.label)
        return len(df)


class PandasSimpleQuery(PandasBaseQuery):
    def apply_selection(self, df, label):
        if label is None:
            return df
        
        if self.column in df.index.names:
            df = df.reset_index()
        if self.column not in df.columns:
            raise KeyError(self.column)
        if isinstance(label, slice):
            if label.step is None:
                ge = df[self.column] >= label.start
                lt = df[self.column] < label.stop
                mask = ge and lt
            else:
                label = list(range(label.start, label.stop, label.step))
        if isinstance(label, list):
            mask = df[self.column].isin(label)
        else:
            mask = df[self.column] == label
        return df.loc[mask]


class PandasIntervalQuery(PandasBaseQuery):

    def as_interval(self, interval):
        if isinstance(interval, tuple):
            left, right = interval
        elif isinstance(interval, dict):
            left, right = interval["left"], interval["right"]
        elif isinstance(interval, slice):
            left, right = interval.start, interval.stop
        elif hasattr(interval, "left") and hasattr(interval, "right"):
            left, right = interval.left, interval.right
        else:
            left = right = interval
        if isinstance(left, datetime):
            left = pd.to_datetime(left)
        if isinstance(right, datetime):
            right = pd.to_datetime(right)

        return pd.Interval(left, right)

    def apply_selection(self, df, label):

        if label is None:
            return df

        if isinstance(label, list):
            intervals = set([self.as_interval(lbl) for lbl in label])
            return pd.concat([self.apply_selection(df, lbl) for lbl in intervals])
        
        interval = self.as_interval(label)

        if self.column in df.index.names:
            df = df.reset_index()
        if self.column not in df.columns:
            raise KeyError(self.column)
        df = df.set_index(self.column)
        
        return df[df.index.overlaps(interval)]


class PandasInterpolationQuery(PandasBaseQuery):
    def apply_selection(self, df, label, limit=1):
        if label is None:
            return df
        
        if isinstance(label, list):
            return pd.concat([self.apply_selection(df, lbl) for lbl in label])

        if self.column in df.index.names:
            df = df.reset_index()

        if self.column not in df.columns:
            raise KeyError(self.column)

        rows = []
        # select all values before requested values
        idx_column = df[self.column]

        if isinstance(label, (datetime, pd.Timestamp)):
            label = pd.to_datetime(label, utc=(idx_column.dt.tz is pytz.UTC))
            label = label.tz_convert(None)
            if idx_column.dt.tz is not None:
                idx_column = idx_column.dt.tz_convert(pytz.UTC)
                idx_column = idx_column.dt.tz_convert(None)
        before = df[idx_column <= label]
        if len(before):
            # if there are values after `value`, we find the closest one
            before = before.sort_values(self.column, ascending=False).head(limit)
            rows.append(before)

        # select all values after requested values
        after = df[idx_column > label]
        if len(after):
            # same as before
            after = after.sort_values(self.column, ascending=True).head(limit)
            rows.append(after)
        if not rows:
            return df.head(0)
        return pd.concat(rows)


class PandasMultiQuery(PandasBaseQuery):
    def __init__(self, index, df, queries: List[PandasBaseQuery]) -> None:
        self.index = index
        self.df = df
        self.queries = queries
        self.label = [q.label for q in queries]

    @property
    def labels(self):
        return {query.column: query.label for query in self.queries}

    def apply_selection(self, df, labels):
        if len(self.queries) == 1:
            return self.queries[0].apply_selection(df, labels[0])

        for query, label in zip(self.queries, labels):
            if isinstance(query, PandasInterpolationQuery):
                selections = []
                others = [q.column for q in self.queries if q is not query]
                if not others:
                    df = query.apply_selection(df, label)
                    continue
                if len(others) == 1:
                    others = others[0]
                for _, pdf in df.groupby(others):
                    selection = query.apply_selection(pdf, label).reset_index()
                    selections.append(selection)

                selections = [s for s in selections if len(s)]
                if not selections:
                    df = df.head(0)
                elif len(selections) == 1:
                    df = selections[0]
                else:
                    df = pd.concat(selections)
            else:
                df = query.apply_selection(df, label)
        return df


@DatasourceInterface.register_interface(pd.DataFrame)
class PandasInterface(DatasourceInterface):
    source: pd.DataFrame

    @classmethod
    def from_url(cls, url: str, **kwargs):
        if url.endswith(".csv"):
            df = pd.read_csv(url, **kwargs)
            return cls(df)
        elif url.endswith(".pq"):
            df = pd.read_parquet(url, **kwargs)
            return cls(df)
        elif url.endswith(".pkl"):
            df = pd.read_pickle(url, **kwargs)
            return cls(df)

        raise NotImplementedError

    @singledispatchmethod
    def compile_query(self, index, label):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {type(index)} indexes."
        )

    @compile_query.register(Index)
    def simple_query(self, index, label):
        return PandasSimpleQuery(index, self.source, index.name, label)

    @compile_query.register(IntervalIndex)
    def interval_query(self, index, label):
        return PandasIntervalQuery(index, self.source, index.name, label)

    @compile_query.register(InterpolatingIndex)
    def interpolating_query(self, index, label):
        return PandasInterpolationQuery(index, self.source, index.name, label)

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, index, labels):
        if not isinstance(index, MultiIndex):
            index = MultiIndex(*index)

        queries = [self.compile_query(idx, labels[idx.name]) for idx in index.indexes]

        return PandasMultiQuery(index, self.source, queries)

    def insert(self, doc):
        index = doc.index_labels_tuple
        index = to_pandas(index)

        if len(index) == 1:
            index = index[0]
        self.source.loc[index, :] = doc.column_values

    def update(self, index_labels, doc):
        index = tuple(to_pandas(index_labels[f]) for f in doc.get_index_fields())
        if len(index) == 1:
            index = index[0]
        self.source.loc[index, :] = doc.column_values

    def delete(self, doc):
        index = doc.index_labels_tuple
        index = to_pandas(index)

        if len(index) == 1:
            index = index[0]
        return self.source.drop(index=index, inplace=True)

    def ensure_index(self, names, order=1):
        index_names = self.source.index.names
        
        if all([name in index_names for name in names]):
            return

        if any([name in index_names for name in names]):
            self.source.reset_index(inplace=True)

        if len(names) == 1:
            names = names[0]
            
        self.source.set_index(names, inplace=True)

    def initdb(self, schema):
        index_names = list(schema.get_index_fields())
        self.ensure_index(index_names)


@singledispatch
def to_pandas(obj):
    return obj


@to_pandas.register(datetime)
def to_pandas_datetime(obj):
    return pd.to_datetime(obj)


@to_pandas.register(dict)
def to_pandas_dict(obj: dict):
    if len(obj) == 2 and "left" in obj and "right" in obj:
        left, right = to_pandas(obj["left"]), to_pandas(obj["right"])
        return pd.Interval(left, right)
    return {k: to_pandas(v) for k, v in obj.items()}


@to_pandas.register(list)
def to_pandas_list(obj):
    return [to_pandas(v) for v in obj]


@to_pandas.register(tuple)
def to_pandas_tuple(obj):
    return tuple(to_pandas(v) for v in obj)


@to_pandas.register(Interval)
def to_pandas_interval(obj):
    left, right = to_pandas(obj.left), to_pandas(obj.right)
    return pd.Interval(left, right)


@singledispatch
def from_pandas(obj):
    return obj


@from_pandas.register(pd.DataFrame)
def from_pandas_df(df):
    return from_pandas(df.to_dict(orient="records"))


@from_pandas.register(pd.Series)
def from_pandas_series(obj):
    return from_pandas(obj.to_dict())


@from_pandas.register(pd.Interval)
def from_pandas_interval(obj):
    left, right = from_pandas(obj.left), from_pandas(obj.right)
    return Interval[left, right]


@from_pandas.register(list)
def from_pandas_list(obj):
    return [from_pandas(v) for v in obj]


@from_pandas.register(tuple)
def from_pandas_tuple(obj):
    return tuple(from_pandas(v) for v in obj)


@from_pandas.register(dict)
def from_pandas_dict(obj):
    return {k: from_pandas(v) for k, v in obj.items()}


@from_pandas.register(pd.Timestamp)
def from_pandas_timestamp(obj):
    return obj.to_pydatetime()


@from_pandas.register(pd.Timedelta)
def from_pandas_timedelta(obj):
    return obj.to_pytimedelta()


@from_pandas.register(numbers.Integral)
def from_pandas_int(obj):
    return int(obj)


@from_pandas.register(numbers.Real)
def from_pandas_float(obj):
    return float(obj)
