from typing import Union

from dephell_versioning import BaseScheme

from ..http_client import BaseHttpClient
from ..indexes import Index, InterpolatingIndex, IntervalIndex, MultiIndex
from ..utils import singledispatchmethod
from .base import BaseDataQuery, DatasourceInterface


class HttpQuery(BaseDataQuery):
    client: BaseHttpClient
    params: dict

    def __init__(self, params=None):
        self.params = params if params is not None else {}

    def apply(self, client):
        return client.find(**self.params)

def serializable_interval(interval):
    if isinstance(interval, list):
        return [serializable_interval(iv) for iv in interval]

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
    
    interval = {'left': left, 'right': right}

    return interval


@DatasourceInterface.register_interface(BaseHttpClient)
class HttpInterface(DatasourceInterface):


    @singledispatchmethod
    def compile_query(self, index, label):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {type(index)} indexes."
        )

    @compile_query.register(InterpolatingIndex)
    @compile_query.register(Index)
    def simple_query(self, index: Union[Index,InterpolatingIndex], label):
        return HttpQuery({index.name: label})

    @compile_query.register(IntervalIndex)
    def interval_query(self, index: IntervalIndex, interval):
        interval = serializable_interval(interval)
        return HttpQuery({index.name: interval})

    @compile_query.register(list)
    @compile_query.register(tuple)
    @compile_query.register(MultiIndex)
    def multi_query(self, indexes, labels):
        if isinstance(indexes, MultiIndex):
            indexes = indexes.indexes
            labels = labels.values()
        
        params = {}
        for idx, label in zip(indexes, labels):
            query = self.compile_query(idx, label)
            if idx.name in query.params:
                params[idx.name] = query.params[idx.name]

        return HttpQuery(params)

    def insert(self, client: BaseHttpClient, doc):
        return client.insert(doc.json())
