import fsspec
import json
from itertools import product
from typing import Any, Dict, List, Optional, Union
from loguru import logger


from .json import from_json, to_json

from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import jsonable, singledispatchmethod, hashable_doc, unhashable_doc

from .base import BaseDataQuery, DatasourceInterface

to_tinydb = to_json
from_tinydb = from_json

from tinydb import TinyDB, Query, where
from tinydb.table import Table
from tinydb.storages import Storage


class FsspecStorage(Storage):
    def __init__(self, path, **storage_kwargs):
        self.path = path
        self.storage_kwargs = storage_kwargs

    def read(self) -> Optional[Dict[str, Dict[str, Any]]]:
        with fsspec.open(self.path, **self.storage_kwargs) as f:
            if not f.size:
                return None
            if not f.readable():
                return None
            data = json.load(f)
            return data

    def write(self, data: Dict[str, Dict[str, Any]]) -> None:
        with fsspec.open(self.path, "rb", **self.storage_kwargs) as f:
            if not f.writable():
                raise IOError(
                    f'Cannot write to the database. Access mode is "{f.mode}"'
                )
            json.dump(data, f)


class TinyDBSelection:
    def __init__(self, query, sort=None, select=None):
        self.query = query

        if isinstance(sort, str):
            sort = (sort,)

        if not isinstance(sort, (tuple, type(None))):
            raise TypeError(f"sort must be string, tuple or None. Got {type(sort)}")

        self.sort = sort
        self.select = select

    def apply(self, db):
        docs = db.search(self.query)

        if self.sort is not None:
            docs = sorted(docs, key=lambda x: tuple(x[s] for s in self.sort))

        if not docs:
            return docs

        if self.select is not None:
            docs = docs[self.select]

        if not isinstance(docs, list):
            docs = [docs]

        return from_tinydb(docs)


class TinyDBQuery(BaseDataQuery):
    def __init__(self, index, labels, table, selections) -> None:
        self.index = index
        self.labels = labels
        self.table = table
        if not isinstance(selections, list):
            selections = [selections]
        self.selections = selections

    def apply_selection(self, db):
        docs = []
        for selection in self.selections:
            selected = selection.apply(self.table)
            docs.extend(selected)
        return docs

    def execute(self, limit=None, skip=None, sort=None):
        logger.debug("Applying tinydb selection")

        docs = self.apply_selection(self.table)

        docs = self.index.reduce(docs, self.labels)

        if not docs:
            return docs

        skip = skip if skip is not None else 0
        limit = limit if limit is not None else len(docs)
        docs = docs[skip:limit]

        return docs

    def unique(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]

        docs = self.apply_selection(self.table)
        results = {}
        for field in fields:
            values = [doc[field] for doc in docs]
            values = set([hashable_doc(v) for v in values])
            values = [unhashable_doc(v) for v in values]
            results[field] = values

        if len(fields) == 1:
            return results[fields[0]]
        return results

    def max(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]

        docs = self.apply_selection(self.table)
        results = {}
        for field in fields:
            values = [doc[field] for doc in docs]
            results[field] = max(values)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def min(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]

        docs = self.apply_selection(self.table)
        results = {}
        for field in fields:
            values = [doc[field] for doc in docs]
            results[field] = min(values)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def count(self):
        docs = self.execute()
        return len(docs)


@DatasourceInterface.register_interface(Table)
@DatasourceInterface.register_interface(TinyDB)
class TinyDBInterface(DatasourceInterface):
    @classmethod
    def from_url(cls, url: str, table: str = None, **kwargs):
        if url.endswith(".json"):
            db = TinyDB(url, storage=FsspecStorage, **kwargs)
            if table is not None:
                db = db.table(table)  # type: ignore
            return cls(db)

        raise NotImplementedError

    @singledispatchmethod
    def compile_query(self, index, label):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {type(index)} indexes."
        )

    @compile_query.register(Index)
    @compile_query.register(str)
    def get_simple_query(self, index, label, others=None):
        logger.debug(
            "Building tinydb simple-query for index: " f"{index} with label: {label}"
        )
        name = index.name if isinstance(index, Index) else index
        target = where(name)

        if isinstance(label, slice):
            # support basic slicing, this will only work
            # for values that are comparable with
            #  the >= and <= operators
            start = jsonable(label.start)
            stop = jsonable(label.stop)
            step = jsonable(label.step)

            if step is None:
                if start is None:
                    query = target.noop()
                else:
                    query = target >= start
                if stop is not None:
                    query = query & (target < stop)
                labels = {index.name: label}
                selection = TinyDBSelection(query)
                return TinyDBQuery(index, labels, self.source, selection)
            else:
                label = list(range(start, stop, step))

        labels = {index.name: label}

        if label is None:
            query = target.noop()
            selection = TinyDBSelection(query)
            return TinyDBQuery(index, labels, self.source, selection)

        label = jsonable(label)

        if isinstance(label, list):
            query = target.one_of(label)
            selection = TinyDBSelection(query)
            return TinyDBQuery(index, labels, self.source, selection)

        if isinstance(label, dict):
            query = target.noop()
            for k, v in label.items():
                query = query & (target[k] == v)
            selection = TinyDBSelection(query)
            return TinyDBQuery(index, labels, self.source, selection)

        query = target == label
        selection = TinyDBSelection(query)
        return TinyDBQuery(index, labels, self.source, selection)

    @compile_query.register(IntervalIndex)
    def interval_query(self, index, label):
        logger.debug(
            "Building tinydb simple-query for index: " f"{index} with label: {label}"
        )
        name = index.name
        target = where(name)

        if not isinstance(label, list):
            label = [label]

        label = jsonable(label)

        queries = []
        for iv in label:
            left, right = extract_interval_edges(iv)
            query = overlap_query(target, left, right)
            queries.append(query)

        query = queries[0]
        for q in queries[1:]:
            query = query | q

        selection = TinyDBSelection(query)
        return TinyDBQuery(index, {index.name: label}, self.source, selection)

    @compile_query.register(InterpolatingIndex)
    def interpolating_query(self, index, label):
        logger.debug(
            "Building tinydb interpolating-query for index: "
            f"{index} with label: {label}"
        )
        name = index.name
        target = where(name)

        label = jsonable(label)

        before_query = target < label
        after_query = target >= label
        before_selection = TinyDBSelection(before_query, sort=name, select=-1)
        after_selection = TinyDBSelection(after_query, sort=name, select=0)
        labels = {index.name: label}
        return TinyDBQuery(
            index, labels, self.source, [before_selection, after_selection]
        )

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, index, labels):
        if not isinstance(index, MultiIndex):
            index = MultiIndex(*index)

        selection_lists = []
        processed_labels = {}
        for idx in index.indexes:
            query = self.compile_query(idx, labels[idx.name])
            processed_labels.update(query.labels)
            selection_lists.append(query.selections)

        selections = []
        for selections_list in product(*selection_lists):
            query = Query().noop()
            sort = ()
            select = None
            for selection in selections_list:
                query = query & selection.query
                if selection.sort is not None:
                    sort = sort + selection.sort
                if selection.select is not None:
                    select = selection.select
            selections.append(TinyDBSelection(query, sort=sort, select=select))

        return TinyDBQuery(index, processed_labels, self.source, selections)

    def insert(self, doc):
        cond = Query().noop()
        for k, v in doc.index_labels.items():
            cond = cond & (where(k) == jsonable(v))
        self.source.remove(cond)
        return self.source.upsert(doc.jsonable(), cond)

    def insert_many(self, docs: list) -> list:
        return self.source.insert_multiple([doc.jsonable() for doc in docs])

    update = insert

    def delete(self, doc):
        cond = Query().noop()
        for k, v in doc.index_labels.items():
            cond = cond & (where(k) == jsonable(v))
        return self.source.remove(cond)

    def initdb(self, schema):
        pass


def overlap_query(target, left, right):
    query = target.noop()

    if left is not None:
        query = query & (target["right"] > left)
    if right is not None:
        query = query & (target["left"] < right)
    return query


def extract_interval_edges(interval):

    # handle different kinds of interval definitions
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

    return left, right
