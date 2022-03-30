
import json
import toolz
import fsspec

from loguru import logger
from datetime import datetime
from typing import Any, List, Union

import numpy as np
import pandas as pd

from .base import BaseDataQuery, DatasourceInterface
from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import jsonable, singledispatchmethod, hashable_doc, unhashable_doc
from ..interpolation import interpolate, interpolate_records

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

    def execute(self, limit: int = None, skip: int = None, sort = None):
        logger.debug('Applying pandas dataframe selection')

        if not len(self.data):
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
            start = skip if skip is not None else 0
            limit = start + limit
            docs = docs[start:limit]
        
        # docs = self.index.reduce(docs, self.labels)
        logger.debug(f'Done. Found {len(docs)} documents.')
        return docs

    def min(self, fields: Union[str,List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        docs = self.apply_selection(self.data)
        results = {}
        for field in fields:
            values = [d[field] for d in docs]
            results[field] = min(values)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def max(self, fields: Union[str,List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        docs = self.apply_selection(self.data)
        results = {}
        for field in fields:
            values = [d[field] for d in docs]
            results[field] = max(values)
        if len(fields) == 1:
            return results[fields[0]]
        return results
    
    def unique(self, fields: Union[str,List[str]]):
            if isinstance(fields, str):
                fields = [fields]
            docs = self.apply_selection(self.data)
            results = {}
            for field in fields:
                values = [doc[field] for doc in docs]
                values = set([hashable_doc(v) for v in values])
                values = [unhashable_doc(v) for v in values]
                results[field] = values

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
    
    def filter(self, record: dict):
        if self.label is None:
            return record
        
        if self.field not in record:
            raise KeyError(self.field)

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

        left, right = jsonable(left), jsonable(right)

        return (record[self.field]['left'] < right) and \
               (record[self.field]['right'] > left)
        

class JsonInterpolationQuery(JsonBaseQuery):
    def apply_selection(self, records):
                
        label = self.labels.get(self.field, None)
        
        if label is None:
            return records

        if not all([self.field in record for record in records]):
            raise KeyError(self.field)

        label = jsonable(label)

        if not isinstance(label, list):
            label = [label]
              
        records = interpolate_records(self.field,
                                      label,
                                      records, 
                                    #   groupby=other_index_names, 
                                      extrapolate=self.index.can_extrapolate(self.labels))
        return records

        # field_values = np.array([record[self.field] for record in records])
        # before_mask = (field_values <= self.label)
        # before_values = field_values[before_mask]

        # after_mask = (field_values > self.label)
        # after_values = field_values[after_mask]

        # before_idx = np.argsort(np.abs(before_values) - self.label)[0]
        # before_records = [records[i] for i in np.flatnonzero(before_mask)]
        # before_value = before_records[before_idx]

        # after_idxs = np.argsort(np.abs(after_values) - self.label)[0]
        # after_records = [records[i] for i in np.flatnonzero(after_mask)]
        # after_values = [after_records[i] for i in after_idxs]
        # return before_values + after_values


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
                
                for _, docs in toolz.groupby(others, records):
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
    def from_url(cls, url: str, jsonpath='', **kwargs):
        if url.endswith(".json"):
            with fsspec.open(url, **kwargs) as f:
                data = json.load(f)
                for p in jsonpath.split('.'):
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
        label = jsonable(label)
        return JsonSimpleQuery(index, self.source, index.name, label)

    @compile_query.register(IntervalIndex)
    def interval_query(self, index, label):
        label = jsonable(label)
        return JsonIntervalQuery(index, self.source, index.name, label)

    @compile_query.register(InterpolatingIndex)
    def interpolating_query(self, index, label):

        # if isinstance(labels, dict) and index.name in labels:
        #     labels = dict(labels)
        # else:
        #     labels = {index.name: labels}

        return JsonInterpolationQuery(index, self.source, index.name, label)

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, index, labels):
        if not isinstance(index, MultiIndex):
            index = MultiIndex(*index)

        queries = [
            self.compile_query(idx, labels[idx.name]) for idx in index.indexes
        ]

        return JsonMultiQuery(index, self.source, queries)

    def _find(self, doc):
        for i, d in enumerate(self.source):
            if doc.same_index(doc.__class__(**d)):
                return i
        else:
            raise KeyError(doc.index_labels)

    def insert(self, doc):
        doc = doc.jsonable()
        self.source.append(doc)

    def update(self, doc):
        for i, d in enumerate(self.source):
            if doc.same_index(doc.__class__(**d)):
                self.source[i] = doc.jsonable()
                break
        else:
            from rframe.schema import UpdateError

            raise UpdateError(f"No document with index {doc.index} found.")

    def delete(self, doc):
        del self.source[self._find(doc)]
