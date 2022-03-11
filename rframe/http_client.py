import requests
from abc import ABC, abstractmethod


class BaseHttpClient(ABC):

    @abstractmethod
    def query(self, **params):
        pass
    
    @abstractmethod
    def insert(self, data):
        pass


class HttpClient(BaseHttpClient):
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers if headers is not None else {}

    def query(self, **params):
        r = requests.get(self.url, headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()
    
    def insert(self, data):
        r = requests.post(self.url, headers=self.headers, data=data)
        r.raise_for_status()
        return r.json()
