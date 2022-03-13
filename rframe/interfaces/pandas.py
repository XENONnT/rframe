from datetime import datetime
from typing import Any, List

import pandas as pd

from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import singledispatchmethod
from .base import BaseDataQuery, DatasourceInterface


class PandasBaseQuery(BaseDataQuery):
    def __init__(self, index, df, column: str, label: Any) -> None:
        self.index = index
        self.df = df
        self.column = column
        self.label = label

    def apply_selection(self, df):
        raise NotImplementedError

    def execute(self, limit: int = None, skip: int = None):
        df = self.apply_selection(self.df)
        if df.index.names:
            df = df.reset_index()
        if limit is not None:
            start = skip * self.index.DOCS_PER_LABEL if skip is not None else 0
            limit = limit * self.index.DOCS_PER_LABEL
            df = df.iloc[start:limit]
        docs = df.to_dict(orient="records")
        labels = {self.column: self.label}
        docs = self.index.reduce(docs, labels)
        return docs

    def min(self, field: str):
        df = self.apply_selection(self.df)
        return df[field].min()

    def max(self, field: str):
        df = self.apply_selection(self.df)
        return df[field].max()
    
    def unique(self, field):
        df = self.apply_selection(self.df)
        return df[field].unique()

class PandasSimpleQuery(PandasBaseQuery):
    def apply_selection(self, df):
        if self.label is None:
            return df
        if self.column in df.index.names:
            df = df.reset_index()
        if self.column not in df.columns:
            raise KeyError(self.column)
        label = self.label
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
    def apply_selection(self, df):
        if self.label is None:
            return df
        if self.column in df.index.names:
            df = df.reset_index()
        if self.column not in df.columns:
            raise KeyError(self.column)
        df = df.set_index(self.column)

        interval = self.label
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
        interval = pd.Interval(left, right)
        return df[df.index.overlaps(interval)]


class PandasInterpolationQuery(PandasBaseQuery):
    def apply_selection(self, df, limit=1):
        if self.label is None:
            return df
        if self.column in df.index.names:
            df = df.reset_index()

        if self.column not in df.columns:
            raise KeyError(self.column)

        rows = []
        # select all values before requested values
        idx_column = df[self.column]
        before = df[idx_column <= self.label]
        if len(before):
            # if ther are values after `value`, we find the closest one
            before = before.sort_values(self.column, ascending=False).head(limit)
            rows.append(before)

        # select all values after requested values
        after = df[idx_column > self.label]
        if len(after):
            # same as before
            after = after.sort_values(self.column, ascending=True).head(limit)
            rows.append(after)

        return pd.concat(rows)


class PandasMultiQuery(PandasBaseQuery):
    def __init__(self, index, df, queries: List[PandasBaseQuery]) -> None:
        self.index = index
        self.df = df
        self.queries = queries

    def apply_selection(self, df):
        for query in self.queries:
            if isinstance(query, PandasInterpolationQuery):
                selections = []
                others = [q.column for q in self.queries if q is not query]
                for _, pdf in df.groupby(others):
                    selections.append(query.apply_selection(pdf).reset_index())
                df = pd.concat(selections)
            else:
                df = query.apply_selection(df)
        return df

    def execute(self, limit: int = None, skip: int = None):
        df = self.apply_selection(self.df)
        if df.index.names:
            df = df.reset_index()
        if limit is not None:
            start = skip * self.index.DOCS_PER_LABEL if skip is not None else 0
            limit = limit * self.index.DOCS_PER_LABEL
            df = df.iloc[start:limit]
        docs = df.to_dict(orient="records")
        labels = {query.column: query.label for query in self.queries}
        docs = self.index.reduce(docs, labels)
        return docs

@DatasourceInterface.register_interface(pd.DataFrame)
class PandasInterface(DatasourceInterface):
    
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

        queries = [
            self.compile_query(idx, labels[idx.name]) for idx in index.indexes
        ]

        return PandasMultiQuery(index, self.source, queries)

    def insert(self, doc):
        index = tuple(doc.index_for(name).to_pandas(label) 
                        for name, label in doc.index_labels.items())
        self.source.loc[index] = doc.column_values