from functools import singledispatch
import json
import fsspec
from rframe.dispatchers import are_equal

from toolz import groupby
from loguru import logger
from typing import Any, List, Union
from pydantic.datetime_parse import datetime_re
from pydantic.validators import parse_datetime

import numpy as np

from ..types import Interval

from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import jsonable, singledispatchmethod, hashable_doc, unhashable_doc
from .base import BaseDataQuery, DatasourceInterface


class JsonBaseQuery(BaseDataQuery):
    def __init__(self, index, data, field: str, label: Any) -> None:
        self.index = index
        self.data = data
        self.field = field
        self.label = label

    @property
    def labels(self):
        return {self.field: self.label}

    def filter(self, record: dict):
        raise NotImplementedError

    def apply_selection(self, records):
        return list(filter(self.filter, records))

    def execute(self, limit: int = None, skip: int = None, sort=None):
        logger.debug("Applying pandas dataframe selection")

        if not self.data:
            return []

        if sort is None:
            data = self.data
        else:
            if isinstance(sort, str):
                sort = [sort]
            data = [hashable_doc(d) for d in self.data]
            data = sorted(data, key=lambda d: tuple(d[s] for s in sort))
            data = [unhashable_doc(d) for d in data]
        docs = self.apply_selection(data)

        if limit is not None:
            start = skip * self.index.DOCS_PER_LABEL if skip is not None else 0
            limit = start + limit * self.index.DOCS_PER_LABEL
            docs = docs[start:limit]

        docs = self.index.reduce(docs, self.labels)

        docs = from_json(docs)

        logger.debug(f"Done. Found {len(docs)} documents.")

        return docs

    def min(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        docs = self.apply_selection(self.data)
        results = {}
        for field in fields:
            values = [d[field] for d in docs]
            results[field] = min(values)
        results = from_json(results)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def max(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        docs = self.apply_selection(self.data)
        results = {}
        for field in fields:
            values = [d[field] for d in docs]
            results[field] = max(values)
        results = from_json(results)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def unique(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        docs = self.apply_selection(self.data)
        results = {}
        for field in fields:
            values = [doc[field] for doc in docs]
            values = set([hashable_doc(v) for v in values])
            values = [unhashable_doc(v) for v in values]
            results[field] = values
        results = from_json(results)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def count(self):
        docs = self.apply_selection(self.data)
        return len(docs)


class JsonSimpleQuery(JsonBaseQuery):
    def filter(self, record: dict):
        if self.label is None:
            return True

        if self.field not in record:
            raise KeyError(self.field)

        label = self.label
        if isinstance(label, slice):
            if label.step is None:
                ge = record[self.field] >= label.start
                lt = record[self.field] < label.stop
                return ge and lt
            else:
                label = list(range(label.start, label.stop, label.step))
        if isinstance(label, list):
            return record[self.field] in label
        else:
            return record[self.field] == label


class JsonIntervalQuery(JsonBaseQuery):
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

        left, right = to_json(left), to_json(right)
        return left, right
    
    def _filter(self, record: dict, label):
        left, right = self.as_interval(label)        

        return (record[self.field]["left"] < right) and (
            record[self.field]["right"] > left
        )
    
    def filter(self, record: dict):
        if self.label is None:
            return record
        
        if isinstance(self.label, list):
            return any([self._filter(record, lbl) for lbl in self.label])
        
        if self.field not in record:
            raise KeyError(self.field)

        return self._filter(record, self.label)


class JsonInterpolationQuery(JsonBaseQuery):

    def _apply_selection(self, records, label, limit=1):
        field_values = np.array([record[self.field] for record in records])
        before_mask = field_values <= label
        before_values = field_values[before_mask]

        after_mask = field_values > label
        after_values = field_values[after_mask]

        before_idxs = np.argsort(np.abs(before_values) - label)[:limit]
        before_records = [records[i] for i in np.flatnonzero(before_mask)]
        before_values = [before_records[i] for i in before_idxs]

        after_idxs = np.argsort(np.abs(after_values) - label)[:limit]
        after_records = [records[i] for i in np.flatnonzero(after_mask)]
        after_values = [after_records[i] for i in after_idxs]
        return before_values + after_values
    
    def apply_selection(self, records, limit=1):
        if self.label is None:
            return records

        if not all(self.field in record for record in records):
            raise KeyError(self.field)

        if isinstance(self.label, list):
            selections = []
            for lbl in self.label:
                selections.extend(self._apply_selection(records, lbl, limit=limit))
            return selections
        
        return self._apply_selection(records, self.label, limit=limit)


class JsonMultiQuery(JsonBaseQuery):
    def __init__(self, index, data, queries: List[JsonBaseQuery]) -> None:
        self.index = index
        self.data = data
        self.queries = queries

    @property
    def labels(self):
        return {query.field: query.label for query in self.queries}

    def apply_selection(self, records):
        if len(self.queries) == 1:
            return self.queries[0].apply_selection(records)

        for query in self.queries:
            if isinstance(query, JsonInterpolationQuery):
                selections = []
                others = [q.field for q in self.queries if q is not query]
                if not others:
                    records = query.apply_selection(records)
                    continue

                for _, docs in groupby(others, records):
                    selection = query.apply_selection(docs).reset_index()
                    selections.extend(selection)

                if selections:
                    records = selections
                else:
                    records = []

            else:
                records = query.apply_selection(records)
        return records


@DatasourceInterface.register_interface(list)
class JsonInterface(DatasourceInterface):
    @classmethod
    def from_url(cls, url: str, jsonpath="", **kwargs):
        if url.endswith(".json"):
            with fsspec.open(url, **kwargs) as f:
                data = json.load(f)
                for p in jsonpath.split("."):
                    data = data[p] if p else data
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain a list of documents")
                return cls(data)
        raise NotImplementedError

    @singledispatchmethod
    def compile_query(self, index, label):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {type(index)} indexes."
        )

    @compile_query.register(Index)
    @compile_query.register(str)
    def simple_query(self, index, label):
        if isinstance(index, str):
            index, name = Index(), index
            index.name = name
        label = to_json(label)
        return JsonSimpleQuery(index, self.source, index.name, label)

    @compile_query.register(IntervalIndex)
    def interval_query(self, index, label):
        label = to_json(label)
        return JsonIntervalQuery(index, self.source, index.name, label)

    @compile_query.register(InterpolatingIndex)
    def interpolating_query(self, index, label):
        label = to_json(label)
        return JsonInterpolationQuery(index, self.source, index.name, label)

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, index, labels):
        if not isinstance(index, MultiIndex):
            index = MultiIndex(*index)

        queries = [self.compile_query(idx, labels[idx.name]) for idx in index.indexes]

        return JsonMultiQuery(index, self.source, queries)

    def _find(self, doc):
        for i, d in enumerate(self.source):
            if doc.same_index(doc.__class__(**d)):
                return i
        else:
            raise KeyError(doc.index_labels)

    def insert(self, doc):
        doc = to_json(doc.dict())
        self.source.append(doc)

    def update(self, index_labels, doc):
        for i, d in enumerate(self.source):
            d = doc.__class__(**d)
            if are_equal(index_labels, d.index_labels):
                self.source[i] = to_json(doc.dict())
                break
        else:
            from rframe.schema import UpdateError

            raise UpdateError(f"No document with index {doc.index} found.")

    def delete(self, doc):
        del self.source[self._find(doc)]

    def initdb(self, schema):
        pass

def to_json(obj):
    return jsonable(obj)


@singledispatch
def from_json(obj):
    return obj


@from_json.register(str)
def from_json_str(obj):
    match = datetime_re.match(obj)  # type: ignore
    if match is None:
        return obj
    return parse_datetime(obj)


@from_json.register(list)
def from_json_list(obj):
    return [from_json(v) for v in obj]


@from_json.register(tuple)
def from_json_tuple(obj):
    return tuple(from_json(v) for v in obj)


@from_json.register(dict)
def from_json_dict(obj):
    if len(obj) == 2 and "left" in obj and "right" in obj:
        left, right = from_json((obj["left"], obj["right"]))
        return Interval[left, right]
    if len(obj) == 3 and "start" in obj and "stop" in obj and "step" in obj:
        return slice(
            from_json(obj["start"]), from_json(obj["stop"]), from_json(obj["step"])
        )
    return {k: from_json(v) for k, v in obj.items()}
