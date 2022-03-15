from loguru import logger
from datetime import datetime
from typing import Any, List, Union

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
    
    @property
    def labels(self):
        return {self.column: self.label}

    def apply_selection(self, df):
        raise NotImplementedError

    def execute(self, limit: int = None, skip: int = None):
        logger.debug('Applying pandas dataframe selection')

        if not len(self.df):
            return []
        df = self.apply_selection(self.df)
       
        if df.index.names or df.index.name:
            df = df.reset_index()
        if limit is not None:
            start = skip * self.index.DOCS_PER_LABEL if skip is not None else 0
            limit = start + limit * self.index.DOCS_PER_LABEL
            df = df.iloc[start:limit]
        docs = df.to_dict(orient="records")
        docs = self.index.reduce(docs, self.labels)
        logger.debug(f'Done. Found {len(docs)} documents.')
        return docs

    def min(self, fields: Union[str,List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        df = self.apply_selection(self.df)
        results = {}
        for field in fields:
            if field in df.index.names:
                df = df.reset_index()
            results[field] = df[field].min()
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def max(self, fields: Union[str,List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        df = self.apply_selection(self.df)
        results = {}
        for field in fields:
            if field in df.index.names:
                df = df.reset_index()
            results[field] = df[field].max()
        if len(fields) == 1:
            return results[fields[0]]
        return results
    
    def unique(self, fields: Union[str,List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        df = self.apply_selection(self.df)
        results = {}
        for field in fields:
            if field in df.index.names:
                df = df.reset_index()
            results[field] = list(df[field].unique())
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def count(self):
        df = self.apply_selection(self.df)
        return len(df)

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
        if not rows:
            return df.head(0)
        return pd.concat(rows)


class PandasMultiQuery(PandasBaseQuery):
    def __init__(self, index, df, queries: List[PandasBaseQuery]) -> None:
        self.index = index
        self.df = df
        self.queries = queries

    @property
    def labels(self):
        return {query.column: query.label for query in self.queries}

    def apply_selection(self, df):
        if len(self.queries) == 1:
            return self.queries[0].apply_selection(df)

        for query in self.queries:
            if isinstance(query, PandasInterpolationQuery):
                selections = []
                others = [q.column for q in self.queries if q is not query]
                if not others:
                    df = query.apply_selection(df)
                    continue

                for _, pdf in df.groupby(others):
                    selection = query.apply_selection(pdf).reset_index()
                    selections.append(selection)

                selections = [s for s in selections if len(s)]
                if not selections:
                    df = df.head(0)
                elif len(selections) == 1:
                    df = selections[0]
                else:
                    df = pd.concat(selections)
            else:
                df = query.apply_selection(df)
        return df


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
        schema = doc.__class__
        index = tuple(schema.index_for(name).to_pandas(label)
                        for name, label in doc.index_labels.items())
        if len(index) == 1:
            index = index[0]
        self.source.loc[index,:] = doc.column_values

    update = insert
