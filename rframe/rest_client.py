import requests

from typing import Mapping

from numpy import isin
from abc import ABC, abstractmethod

from .utils import jsonable


class BaseRestClient(ABC):

    @abstractmethod
    def query(self, **params):
        pass
    
    @abstractmethod
    def insert(self, data):
        pass


class RestClient(BaseRestClient):
    def __init__(self, url, headers=None, client=None):
        self.url = url
        self.headers = headers if headers is not None else {}
        if client is None:
            client = requests
        self.client = client

    def query(self, limit: int = None, skip: int = None, **params):
        params = jsonable(params)
        r = self.client.post(self.url, headers=self.headers,
                        json=params, params={'limit': limit, 'skip': skip})
        r.raise_for_status()
        return r.json()

    def insert(self, doc):
        doc = jsonable(doc)
        r = self.client.put(self.url, headers=self.headers, json=doc)
        r.raise_for_status()
        return r.json()
