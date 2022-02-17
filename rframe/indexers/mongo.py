
from ast import Import
from warnings import warn
from .base import BaseDataQuery, DatasourceIndexer
from ..utils import singledispatchmethod
from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex


try:
    import pymongo
    from pymongo.collection import Collection


    class MongoAggregation(BaseDataQuery):
        pipeline: list

        def __init__(self, pipeline):
            self.pipeline = pipeline
    
        def apply(self, collection: Collection):
            if not isinstance(collection, Collection):
                raise TypeError(f'collection must be a pymongo Collection, got {type(collection)}.')

            return list(collection.aggregate(self.pipeline))

        def logical_and(self, other):
            return MongoAggregation(self.pipeline+other.pipeline)

        def __add__(self, other):
            return self.logical_and(other)

        def __and__(self, other):
            return self.logical_and(other)


    @DatasourceIndexer.register_indexer(pymongo.collection.Collection)
    class MongoIndexer(DatasourceIndexer):

        @singledispatchmethod
        def compile_query(self, index, label):
            raise NotImplementedError(f'{self.__class__.__name__} does not support {type(index)} indexes.')
        
        @compile_query.register(list)
        @compile_query.register(tuple)
        @compile_query.register(MultiIndex)
        def multi_query(self, index, labels):
            if isinstance(index, MultiIndex):
                index = index.indexes
                labels = labels.values()
            agg = MongoAggregation([])
            for idx,label in zip(index, labels):
                agg = agg + self.compile_query(idx, label)
            return agg

        @compile_query.register(Index)
        @compile_query.register(str)
        def get_simple_query(self, index, label):
            name = index.name if isinstance(index, Index) else index

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
                        label['$gte'] = start
                    if stop is not None:
                        label['$lt'] = stop
                    if not label:
                        label = None
                else:
                    label = list(range(start, stop, step))

            if isinstance(label, list):
                # support querying multiple values
                # in the same request
                label = {'$in': label}

            pipeline = []
            if label is not None:
                pipeline.append({'$match': {name: label} })

            return MongoAggregation(pipeline)

        @compile_query.register(InterpolatingIndex)
        def build_interpolation_query(self, index, label):
            '''For interpolation we match the values directly before and after
            the value of interest. For each value we take the closest document on either side.
            '''
            if label is None:
                return MongoAggregation([])

            if not isinstance(label, list):
                label = [label]

            queries = {f'agg{i}': mongo_closest_query(index.name, value) for i,value in enumerate(label)}
            pipeline = [
                    {
                        # support multiple independent aggregations
                        # using the facet feature 
                        '$facet': queries,
                    },

                    # Combine results of all aggregations 
                    {
                        '$project': {
                            'union': {
                                '$setUnion': [f'${name}' for name in queries],
                                }
                                }
                    },
                    # we just want a single list of documents
                    {
                        '$unwind': '$union'
                    },
                    # move list of documents to the root of the result
                    # so we just get a nice list of documents
                    {
                        '$replaceRoot': { 'newRoot': "$union" }
                    },
                ]
            return MongoAggregation(pipeline)     

        @compile_query.register(IntervalIndex)
        def build_interval_query(self, index, intervals):
            '''Query overlaping documents with given interval, supports multiple 
            intervals as well as zero length intervals (left==right)
            multiple overlap queries are joined with the $or operator 
            '''
            if not isinstance(intervals, list):
                intervals = [intervals]

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
                    '$match':  {
                        # support querying for multiple values
                        # in a single pipeline
                        '$or': queries,
                    }
                }
                ]
            else:
                pipeline = []
            
            return MongoAggregation(pipeline)

        def insert(self, collection: Collection, doc):
            '''We want the client logic to be agnostic to
            whether the value being replaced is actually stored in the DB or
            was inferred from e.g interpolation. 
            The find_one_and_replace(upsert=True) logic is the best match
            for the behavior we want even though it wasts an insert operation
            when a document already exists.
            FIXME: Maybe we can optimize this with an pipeline to
            avoid replacing existing documents with a copy.
            '''
            from rframe.schema import InsertionError

            try:
                doc = collection.find_one_and_update(doc.index_labels,
                            {'$set': doc.dict()}, projection={'_id': False},
                            upsert=True, return_document=pymongo.ReturnDocument.AFTER)
                return doc
            except Exception as e:
                raise InsertionError(f"Mongodb has rejected this insertion:\n {e} ")

except ImportError:
    warn('Pymongo not found, cannot register mongodb interface')

def mongo_overlap_query(index, interval):
    '''Builds a single overlap query
    Intervals with one side equal to null are treated as extending to infinity in
    that direction.
    Supports closed or open intervals as well as infinite intervals

    Overlap definition:
    The two intervals (L,R) and (l,r) overlap iff L<r and l<R
    The operator < is replaced with <= when the interval is closed on that side.
    Where if L/l are None, they are treated as -inf
    and if R/r are None, the are treated as inf 
    '''

    # Set the appropriate operators depending on if the interval
    # is closed on one side or both
    closed = getattr(index, 'closed', 'right')
    gt_op = '$gte' if closed in ['right', 'both'] else '$gt'
    lt_op = '$lte' if closed in ['left', 'both'] else '$lt'

    # handle different kinds of interval definitions
    if isinstance(interval, tuple):
        left, right = interval
    elif isinstance(interval, slice):
        left, right = interval.start, interval.stop
    else:
        left = right = interval
    
    # Some conditions may not apply if the query interval is None
    # on one or both sides
    conditions = []
    if left is not None:
        conditions.append(
                {
                    '$or': [
                        # if the right side of the queried interval is
                        # None, treat it as inf
                        {f'{index.name}.right': None},
                        {f'{index.name}.right': {gt_op: left}},
                        ]
                }
            )
    if right is not None:
        conditions.append(
                {
                    '$or': [{f'{index.name}.left': None},
                            {f'{index.name}.left': {lt_op: right}}]
                }
            )
    if conditions:
        return {
                '$and': conditions,
            }
    else:
        return {}

def mongo_closest_query(name, value):
    return [
        {
            '$addFields': {
                # Add a field splitting the documents into
                # before and after the value of interest
                '_after': {'$gte': [f'${name}', value]},

                # Add a field with the distance to the value of interest
                '_diff': {'$abs': {'$subtract': [value, f'${name}']}},        
                }
        },
        {
            # sort in ascending order by distance
            '$sort': {'_diff': 1},
        },
        {
            # first group by whether document is before or after the value
            # the take the first document in each group
            '$group' : { '_id' : '$_after', 'doc': {'$first': '$$ROOT'},  }
        },
        {
            # make the documents the new root, discarding the groupby value
            "$replaceRoot": { "newRoot": "$doc" },
        },
        {
            # drop the extra fields, they are no longer needed
            '$project': {'_diff':0, '_after':0 },
        },
    ]
