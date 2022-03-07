import requests


class BaseHttpClient:
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers if headers is not None else {}

    def find(self, **params):
        r = requests.get(self.url, headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()

    def insert(self, data):
        r = requests.post(self.url, headers=self.headers, data=data)
        r.raise_for_status()
        return r.json()
