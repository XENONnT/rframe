import datetime
import pymongo
import numbers

from functools import singledispatch
from itertools import product
from typing import List, Union
from loguru import logger

import pandas as pd
from pydantic import BaseModel

from pymongo.collection import Collection
from ..types import Interval

from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import singledispatchmethod
from .base import BaseDataQuery, DatasourceInterface


query_precendence = {
    Index: 1,
    IntervalIndex: 2,
    InterpolatingIndex: 3,
}


class MultiMongoAggregation(BaseDataQuery):
    aggregations: list

    def __init__(self, collection: Collection, aggregations: list):
        if not isinstance(collection, Collection):
            raise TypeError(
                f"collection must be a pymongo Collection, got {type(collection)}."
            )
        self.collection = collection
        self.aggregations = aggregations

    def execute(self, limit=None, skip=None):
        logger.debug("Executing multi mongo aggregation.")
        results = []
        for agg in self.aggregations:
            results.extend(agg.execute(self.collection, limit=limit, skip=skip))
        skip = 0 if skip is None else skip

        if limit is not None:
            return results[skip : limit + skip]

        return results

    def logical_or(self, other: "MultiMongoAggregation"):
        if isinstance(other, MultiMongoAggregation):
            extra = other.aggregations
        else:
            extra = [other]

        return MultiMongoAggregation(self.collection, self.aggregations + extra)

    def __or__(self, other):
        return self.logical_or(other)

    def __add__(self, other):
        return self.logical_or(other)


class MongoAggregation(BaseDataQuery):
    pipeline: list

    def __init__(self, index, labels, collection: Collection, pipeline: list):
        if not isinstance(collection, Collection):
            raise TypeError(
                f"collection must be a pymongo Collection, got {type(collection)}."
            )
        self.index = index
        self.labels = labels
        self.collection = collection
        self.pipeline = pipeline

    @property
    def docs_per_label(self):
        n = 1
        if isinstance(self.index, MultiIndex):
            for index in self.index.indexes:
                if self.labels.get(index.name, None) is not None:
                    n *= index.DOCS_PER_LABEL
        elif self.labels.get(self.index.name, None) is not None:
            n *= self.index.DOCS_PER_LABEL
        return n

    def execute(self, limit: int = None, skip: int = None, sort=None):
        return list(self.iter(limit=limit, skip=skip, sort=sort))

    def iter(self, limit=None, skip=None, sort=None):
        pipeline = list(self.pipeline)

        if sort is None:
            sort = self.index.names

        sort = [sort] if isinstance(sort, str) else sort
        if isinstance(sort, list):
            sort_arg = {field: 1 for field in sort}
        elif isinstance(sort, dict):
            sort_arg = sort
        else:
            raise TypeError(f"sort must be a list or dict, got {type(sort)}.")
        pipeline = pipeline + [{"$sort": sort_arg}]

        if isinstance(skip, int):
            raw_skip = skip * self.docs_per_label
            pipeline.append({"$skip": raw_skip})

        if isinstance(limit, int):
            raw_limit = limit * self.docs_per_label
            raw_limit = int(raw_limit)
            pipeline.append({"$limit": raw_limit})

        pipeline.append({"$project": {"_id": 0}})

        logger.debug(f"Executing mongo aggregation: {pipeline}.")

        # docs = list(self.collection.aggregate(pipeline, allowDiskUse=True))
        collected = 0
        limit = limit if limit is not None else float("inf")
        docs = []
        for doc in self.collection.aggregate(pipeline, allowDiskUse=True):
            docs.append(doc)
            if len(docs) >= self.docs_per_label:
                docs = self.index.reduce(docs, self.labels)
                for doc in docs:
                    yield from_mongo(doc)
                    collected += 1
                    if collected >= limit:
                        return
                docs = []
        if len(docs) and collected < limit:
            docs = self.index.reduce(docs, self.labels)
            for doc in docs:
                yield from_mongo(doc)
                collected += 1
                if collected >= limit:
                    return

    def unique(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        results = {}
        for field in fields:
            pipeline = list(self.pipeline)
            pipeline.append(
                {
                    "$group": {
                        "_id": "$" + field,
                        "first": {"$first": "$" + field},
                    }
                }
            )

            results[field] = [
                doc["first"]
                for doc in self.collection.aggregate(pipeline, allowDiskUse=True)
            ]

        results = from_mongo(results)

        if len(fields) == 1:
            return results[fields[0]]
        return results

    def max(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        results = {}
        for field in fields:
            pipeline = list(self.pipeline)
            pipeline.append({"$sort": {field: -1}})
            pipeline.append({"$limit": 1})
            pipeline.append({"$project": {"_id": 0}})
            try:

                results[field] = next(
                    self.collection.aggregate(pipeline, allowDiskUse=True)
                )[field]
            except (StopIteration, KeyError):
                results[field] = None

        results = from_mongo(results)

        if len(fields) == 1:
            return results[fields[0]]
        return results

    def min(self, fields: Union[str, List[str]]):
        if isinstance(fields, str):
            fields = [fields]
        results = {}
        for field in fields:
            pipeline = list(self.pipeline)
            pipeline.append({"$sort": {field: 1}})
            pipeline.append({"$limit": 1})
            pipeline.append({"$project": {"_id": 0}})
            try:
                results[field] = next(
                    self.collection.aggregate(pipeline, allowDiskUse=True)
                )[field]
            except (StopIteration, KeyError):
                results[field] = None

        results = from_mongo(results)

        if len(fields) == 1:
            return results[fields[0]]
        return results

    def count(self):
        pipeline = list(self.pipeline)
        pipeline.append({"$count": "count"})
        try:
            result = next(self.collection.aggregate(pipeline, allowDiskUse=True))
        except StopIteration:
            return 0
        return result.get("count", 0)

    def logical_and(self, other):
        index = MultiIndex(self.index, other.index)
        labels = dict(self.labels, **other.labels)
        return MongoAggregation(
            index, labels, self.collection, self.pipeline + other.pipeline
        )

    def logical_or(self, other):
        if isinstance(other, MongoAggregation):
            return MultiMongoAggregation(self.collection, [self, other])

        if isinstance(other, MultiMongoAggregation):
            return other + self

    def __add__(self, other):
        return self.logical_or(other)

    def __and__(self, other):
        return self.logical_and(other)

    def __mul__(self, other):
        return self.logical_and(other)


@DatasourceInterface.register_interface(pymongo.collection.Collection)
class MongoInterface(DatasourceInterface):
    @classmethod
    def from_url(
        cls, source: str, database: str = None, collection: str = None, **kwargs
    ):
        if source.startswith("mongodb"):
            if database is None:
                raise ValueError("database must be specified")
            if collection is None:
                raise ValueError("collection must be specified")
            source = pymongo.MongoClient(source)[database][collection]
            return cls(source)

        raise NotImplementedError

    @singledispatchmethod
    def compile_query(self, index, label):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {type(index)} indexes."
        )

    def simple_multi_query(self, index, labels):
        pipeline = []
        indexes = sorted(index.indexes, key=lambda x: query_precendence[type(x)])
        for idx in indexes:
            label = labels[idx.name]
            if label is None:
                continue
            others = [name for name in index.names if name != idx.name]
            agg = self.compile_query(idx, label, others=others)
            pipeline.extend(agg.pipeline)
        return MongoAggregation(index, labels, self.source, pipeline)

    def product_multi_query(self, index, labels):
        labels = [label if isinstance(label, list) else [label] for label in labels]
        aggs = []
        for label_vals in product(*labels):
            agg = self.simple_multi_query(index.indexes, label_vals)
            aggs.append(agg)
        return MultiMongoAggregation(self.source, aggs)

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, index, labels):
        logger.debug(
            "Building mongo multi-query for index: " f"{index} with labels: {labels}"
        )
        if not isinstance(index, MultiIndex):
            index = MultiIndex(*index)
        return self.simple_multi_query(index, labels)

    @compile_query.register(Index)
    @compile_query.register(str)
    def get_simple_query(self, index, label, others=None):
        logger.debug(
            "Building mongo simple-query for index: " f"{index} with label: {label}"
        )
        name = index.name if isinstance(index, Index) else index

        label = to_mongo(label)

        if isinstance(label, slice):
            # support basic slicing, this will only work
            # for values that are comparable with the
            #  $gt/$lt operators
            start = label.start
            stop = label.stop
            step = label.step
            if step is None:
                label = {}
                if start is not None:
                    label["$gte"] = start
                if stop is not None:
                    label["$lt"] = stop
                if not label:
                    label = None
            else:
                label = list(range(start, stop, step))

        match = {name: label}

        if isinstance(label, list):
            # support querying multiple values
            # in the same request
            match = {name: {"$in": label}}

        elif isinstance(label, dict):
            match = {f"{name}.{k}": v for k, v in label.items()}

        pipeline = []
        if label is not None:
            pipeline.append({"$match": match})

        labels = {name: label}

        return MongoAggregation(index, labels, self.source, pipeline)

    @compile_query.register(InterpolatingIndex)
    def build_interpolation_query(self, index, label, others=None):
        """For interpolation we match the values directly before and after
        the value of interest. For each value we take the closest document on either side.
        """
        logger.debug(
            "Building mongo interpolating-query for index: "
            f"{index} with label: {label}"
        )

        label = to_mongo(label)

        if label is None:
            labels = {index.name: label}
            return MongoAggregation(index, labels, self.source, [])
        limit = 1 if others is None else others

        if not isinstance(label, list):
            pipelines = dict(
                before=mongo_before_query(index.name, label, limit=limit),
                after=mongo_after_query(index.name, label, limit=limit),
            )
            pipeline = merge_pipelines(pipelines)
            labels = {index.name: label}
            return MongoAggregation(index, labels, self.source, pipeline)

        pipelines = {
            f"agg{i}": mongo_closest_query(index.name, value)
            for i, value in enumerate(label)
        }
        pipeline = merge_pipelines(pipelines)

        labels = {index.name: label}

        return MongoAggregation(index, labels, self.source, pipeline)

    @compile_query.register(IntervalIndex)
    def build_interval_query(self, index, label, others=None):
        """Query overlaping documents with given interval, supports multiple
        intervals as well as zero length intervals (left==right)
        multiple overlap queries are joined with the $or operator
        """

        logger.debug(
            "Building mongo interval-query for index: " f"{index} with label: {label}"
        )

        if isinstance(label, list):
            intervals = label
        else:
            intervals = [label]

        intervals = to_mongo(intervals)

        queries = []
        for interval in intervals:
            if interval is None:
                continue
            if isinstance(interval, tuple) and all([i is None for i in interval]):
                continue

            query = mongo_overlap_query(index, interval)
            if query:
                queries.append(query)

        if queries:
            pipeline = [
                {
                    "$match": {
                        # support querying for multiple values
                        # in a single pipeline
                        "$or": queries,
                    }
                },
            ]
        else:
            pipeline = []

        labels = {index.name: intervals}

        return MongoAggregation(index, labels, self.source, pipeline)

    def insert(self, doc):
        """We want the client logic to be agnostic to
        whether the value being replaced is actually stored in the DB or
        was inferred from e.g interpolation.
        The find_one_and_replace(upsert=True) logic is the best match
        for the behavior we want even though it wasts an insert operation
        when a document already exists.
        FIXME: Maybe we can optimize this with an pipeline to
        avoid replacing existing documents with a copy.
        """
        from rframe.schema import InsertionError

        index = to_mongo(doc.index_labels)
        try:
            doc = self.source.find_one_and_update(
                index,
                {"$set": to_mongo(doc.dict())},
                projection={"_id": False},
                upsert=True,
                return_document=pymongo.ReturnDocument.AFTER,
            )
            return doc
        except Exception as e:
            raise InsertionError(f"Mongodb has rejected this insertion:\n {e} ")

    update = insert

    def ensure_index(self, names, order=pymongo.ASCENDING):
        self.source.ensure_index([(name, order) for name in names])

    def delete(self, doc):
        index = to_mongo(doc.index_labels)
        return self.source.delete_one(index)

    def initdb(self, schema):
        index_names = list(schema.get_index_fields())
        self.ensure_index(index_names)


def mongo_overlap_query(index, interval):
    """Builds a single overlap query
    Intervals with one side equal to null are treated as extending to infinity in
    that direction.
    Supports closed or open intervals as well as infinite intervals

    Overlap definition:
    The two intervals (L,R) and (l,r) overlap iff L<r and l<R
    The operator < is replaced with <= when the interval is closed on that side.
    Where if L/l are None, they are treated as -inf
    and if R/r are None, the are treated as inf
    """

    # Set the appropriate operators depending on if the interval
    # is closed on one side or both
    closed = getattr(index, "closed", "right")
    gt_op = "$gte" if closed == "both" else "$gt"
    lt_op = "$lte" if closed == "both" else "$lt"

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

    # Some conditions may not apply if the query interval is None
    # on one or both sides
    conditions = []
    if left is not None:
        conditions.append(
            {
                "$or": [
                    # if the right side of the queried interval is
                    # None, treat it as inf
                    {f"{index.name}.right": None},
                    {f"{index.name}.right": {gt_op: left}},
                ]
            }
        )
    if right is not None:
        conditions.append(
            {
                "$or": [
                    {f"{index.name}.left": None},
                    {f"{index.name}.left": {lt_op: right}},
                ]
            }
        )
    if conditions:
        return {
            "$and": conditions,
        }
    else:
        return {}


def mongo_before_query(name, value, limit=1):
    if isinstance(limit, list):
        return mongo_grouped_before_query(name, value, limit)
    return [
        {"$match": {f"{name}": {"$lte": value}}},
        {"$sort": {f"{name}": -1}},
        {"$limit": limit},
    ]


def mongo_after_query(name, value, limit=1):
    if isinstance(limit, list):
        return mongo_grouped_after_query(name, value, limit)
    return [
        {"$match": {f"{name}": {"$gt": value}}},
        {"$sort": {f"{name}": 1}},
        {"$limit": limit},
    ]


def mongo_grouped_before_query(name, value, groups):

    return [
        {"$match": {f"{name}": {"$lte": value}}},
        {"$sort": {f"{name}": -1}},
        {"$group": {"_id": [f"${grp}" for grp in groups], "doc": {"$first": "$$ROOT"}}},
        {
            # make the documents the new root, discarding the groupby value
            "$replaceRoot": {"newRoot": "$doc"},
        },
    ]


def mongo_grouped_after_query(name, value, groups):
    return [
        {"$match": {f"{name}": {"$gt": value}}},
        {"$sort": {f"{name}": 1}},
        {"$group": {"_id": [f"${grp}" for grp in groups], "doc": {"$first": "$$ROOT"}}},
        {
            # make the documents the new root, discarding the groupby value
            "$replaceRoot": {"newRoot": "$doc"},
        },
    ]


def mongo_closest_query(name, value):
    return [
        {
            "$addFields": {
                # Add a field splitting the documents into
                # before and after the value of interest
                "_after": {"$gte": [f"${name}", value]},
                # Add a field with the distance to the value of interest
                "_diff": {"$abs": {"$subtract": [value, f"${name}"]}},
            }
        },
        {
            # sort in ascending order by distance
            "$sort": {"_diff": 1},
        },
        {
            # first group by whether document is before or after the value
            # the take the first document in each group
            "$group": {
                "_id": "$_after",
                "doc": {"$first": "$$ROOT"},
            }
        },
        {
            # make the documents the new root, discarding the groupby value
            "$replaceRoot": {"newRoot": "$doc"},
        },
        {
            # drop the extra fields, they are no longer needed
            "$project": {"_diff": 0, "_after": 0},
        },
    ]


def merge_pipelines(pipelines):
    pipeline = [
        {
            # support multiple independent aggregations
            # using the facet feature
            "$facet": pipelines,
        },
        # Combine results of all aggregations
        {
            "$project": {
                "union": {
                    "$setUnion": [f"${name}" for name in pipelines],
                }
            }
        },
        # we just want a single list of documents
        {"$unwind": "$union"},
        # move list of documents to the root of the result
        # so we just get a nice list of documents
        {"$replaceRoot": {"newRoot": "$union"}},
    ]
    return pipeline


@singledispatch
def to_mongo(obj):
    return obj


@to_mongo.register(dict)
def to_mongo_dict(obj: dict):
    return {k: to_mongo(v) for k, v in obj.items()}


@to_mongo.register(list)
def to_mongo_list(obj):
    return [to_mongo(v) for v in obj]


@to_mongo.register(tuple)
def to_mongo_tuple(obj):
    return tuple(to_mongo(v) for v in obj)


@to_mongo.register(BaseModel)
def to_mongo_interval(obj):
    return to_mongo(obj.dict())


@to_mongo.register(pd.DataFrame)
def to_mongo_df(df):
    return to_mongo(df.to_dict(orient="records"))


@to_mongo.register(datetime.datetime)
def to_mongo_datetime(obj):
    # mongodb datetime has millisecond resolution
    return obj.replace(microsecond=int(obj.microsecond / 1000) * 1000)


@to_mongo.register(datetime.timedelta)
def to_mongo_timedelta(obj):
    # mongodb datetime has millisecond resolution
    seconds = int(obj.total_seconds() * 1e3) / 1e3
    return datetime.timedelta(seconds=seconds)


@to_mongo.register(pd.Timestamp)
def to_mongo_timestamp(obj):
    return to_mongo(obj.to_pydatetime())


@to_mongo.register(pd.Timedelta)
def to_mongo_pdtimedelta(obj):
    return to_mongo(obj.to_pytimedelta())


@to_mongo.register(numbers.Integral)
def to_mongo_int(obj):
    return int(obj)


@to_mongo.register(numbers.Real)
def to_mongo_float(obj):
    return float(obj)


@singledispatch
def from_mongo(obj):
    return obj


@from_mongo.register(list)
def from_mongo_list(obj):
    return [from_mongo(v) for v in obj]


@from_mongo.register(tuple)
def from_mongo_tuple(obj):
    return tuple(from_mongo(v) for v in obj)


@from_mongo.register(dict)
def from_mongo_dict(obj):
    if len(obj) == 2 and "left" in obj and "right" in obj:
        return Interval[obj["left"], obj["right"]]
    return {k: from_mongo(v) for k, v in obj.items()}
