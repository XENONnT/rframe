from typing import Mapping
from pydantic import BaseModel
import requests
from abc import ABC, abstractmethod

from pydantic.json import ENCODERS_BY_TYPE

class BaseHttpClient(ABC):

    @abstractmethod
    def query(self, **params):
        pass
    
    @abstractmethod
    def insert(self, data):
        pass

def jsonable(obj):
    if isinstance(obj, BaseModel):
        return obj.dict()

    if isinstance(obj, Mapping):
        return {k: jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, set, frozenset, tuple)):
        return [jsonable(v) for v in obj]

    if isinstance(obj, (str, int, float, type(None))):
        return obj

    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)
    
    raise TypeError(f"Cannot convert {type(obj)} to JSON")

class HttpClient(BaseHttpClient):
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers if headers is not None else {}

    def query(self, limit: int = None, skip: int = None, **params):
        params = jsonable(params)
        r = requests.post(self.url, headers=self.headers,
                        json=params, params={'limit': limit, 'skip': skip})
        r.raise_for_status()
        return r.json()

    def insert(self, doc):
        doc = jsonable(doc)
        r = requests.put(self.url, headers=self.headers, json=doc)
        r.raise_for_status()
        return r.json()
