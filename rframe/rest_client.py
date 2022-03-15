import requests

from loguru import logger
from typing import List, Mapping

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

    def unique(self, field):
        raise NotImplementedError
    
    def max(self, field):
        raise NotImplementedError
    
    def min(self, field):
        raise NotImplementedError


class RestClient(BaseRestClient):
    def __init__(self, url, headers=None, client=None):
        self.url = url
        self.headers = headers if headers is not None else {}
        if client is None:
            client = requests
        self.client = client

    @property
    def summary_url(self):
        return self.url.rstrip('/') + '/summary'

    def query(self, limit: int = None, skip: int = None, **labels):
        labels = jsonable(labels)
        params = {}
        if limit is not None:
            params['limit'] = limit
        if skip is not None:
            params['skip'] = skip
        r = self.client.post(self.url, headers=self.headers,
                        json=labels, params=params)
        
        with logger.catch():
            r.raise_for_status()
        return r.json()

    def insert(self, doc):
        data = doc.json()
        r = self.client.put(self.url, headers=self.headers, data=data)
        with logger.catch():
            r.raise_for_status()
        return r.json()

    def unique(self, fields: List[str] = None, **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {'fields': fields, 'unique': True}
        r = self.client.post(self.summary_url, headers=self.headers,
                            params=params, json=labels)
        with logger.catch():
            r.raise_for_status()
        data = r.json()
        
        results = {field: data[field]['unique'] for field in fields}
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def max(self, fields: List[str] = None, **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {'fields': fields, 'max': True}
        r = self.client.post(self.summary_url, headers=self.headers,
                            params=params, json=labels)

        with logger.catch():
            r.raise_for_status()
        data = r.json()

        results = {field: data[field]['max'] for field in fields}
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def min(self, fields: List[str] = None, **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {'fields': fields, 'min': True}
        r = self.client.post(self.summary_url, headers=self.headers, 
                            params=params, json=labels)

        with logger.catch():
            r.raise_for_status()
        data = r.json()

        results = {field: data[field]['min'] for field in fields}
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def count(self, **labels):
        labels = jsonable(labels)
        params = {'count': 'true'}
        r = self.client.post(self.summary_url, headers=self.headers, 
                            params=params, json=labels)
                            
        with logger.catch():
            r.raise_for_status()
        data = r.json()
        cnt = data.get('count', None)
        if cnt is None:
            raise ValueError('Failed to fetch count from server.')
        return int(cnt)
    
    def summary(self, fields: List[str] = None, **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {'fields': fields, 'min': True, 'max': True,
                'unique': True, 'count': True}
        r = self.client.post(self.summary_url, headers=self.headers,
                            params=params, json=labels)

        with logger.catch():
            r.raise_for_status()
        data = r.json()

        if len(fields) == 1:
            return data[fields[0]]
        return data