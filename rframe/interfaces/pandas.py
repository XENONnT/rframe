from datetime import datetime
from typing import Any, List

import pandas as pd

from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import singledispatchmethod
from .base import BaseDataQuery, DatasourceInterface


class PandasBaseQuery(BaseDataQuery):
    def __init__(self, column: str, label: Any) -> None:
        super().__init__()
        self.column = column
        self.label = label

    def apply_selection(self, df):
        raise NotImplementedError

    def apply(self, df):
        df = self.apply_selection(df)
        if df.index.names:
            df = df.reset_index()
        return df.to_dict(orient="records")


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
    def __init__(self, queries: List[PandasBaseQuery]) -> None:
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


@DatasourceInterface.register_interface(pd.DataFrame)
class PandasInterface(DatasourceInterface):
    @singledispatchmethod
    def compile_query(self, index, label):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {type(index)} indexes."
        )

    @compile_query.register(Index)
    def simple_query(self, index, label):
        return PandasSimpleQuery(index.name, label)

    @compile_query.register(IntervalIndex)
    def interval_query(self, index, label):
        return PandasIntervalQuery(index.name, label)

    @compile_query.register(InterpolatingIndex)
    def interpolating_query(self, index, label):
        return PandasInterpolationQuery(index.name, label)

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, indexes, labels):
        if isinstance(indexes, MultiIndex):
            indexes = indexes.indexes
            labels = labels.values()

        queries = [
            self.compile_query(idx, label) for idx, label in zip(indexes, labels)
        ]

        return PandasMultiQuery(queries)
