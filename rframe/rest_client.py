import requests

from loguru import logger
from typing import List, Mapping

from numpy import isin
from abc import ABC, abstractmethod

from .utils import jsonable
from .interfaces.json import from_json


class BaseRestClient(ABC):
    @abstractmethod
    def query(self, **params):  # pragma: no cover
        pass

    @abstractmethod
    def insert(self, doc):  # pragma: no cover
        pass

    def unique(self, fields: List[str]):
        raise NotImplementedError

    def max(self, fields: List[str]):
        raise NotImplementedError

    def min(self, fields: List[str]):
        raise NotImplementedError


class RestClient(BaseRestClient):
    QUERY_PATH = "/query"
    INSERT_PATH = "/insert"
    DELETE_PATH = "/delete"
    SUMMARY_PATH = "/summary"

    def __init__(self, url, headers=None, client=None, auth=None):
        self.base_url = url
        self.headers = headers if headers is not None else {}
        self.auth = auth

        if client is None:
            client = requests
        self.client = client

    @property
    def summary_url(self):
        return self.base_url.rstrip("/") + self.SUMMARY_PATH

    @property
    def query_url(self):
        return self.base_url.rstrip("/") + self.QUERY_PATH

    @property
    def insert_url(self):
        return self.base_url.rstrip("/") + self.INSERT_PATH

    @property
    def delete_url(self):
        return self.base_url.rstrip("/") + self.DELETE_PATH

    def query(self, limit: int = None, skip: int = None, sort=None, **labels):
        labels = jsonable(labels)
        params = {}
        if limit is not None:
            params["limit"] = limit
        if skip is not None:
            params["skip"] = skip
        if sort is not None:
            params["sort"] = sort
        r = self.client.post(
            self.query_url,
            headers=self.headers,
            json=labels,
            params=params,
            auth=self.auth,
        )

        with logger.catch():
            r.raise_for_status()
        data = r.json()
        data = from_json(data)
        return data

    def insert(self, doc):
        data = doc.json()
        r = self.client.put(
            self.insert_url, headers=self.headers, data=data, auth=self.auth
        )
        with logger.catch():
            r.raise_for_status()
        return r.json()

    update = insert

    def delete(self, doc):
        data = doc.json()
        r = self.client.delete(
            self.delete_url, headers=self.headers, data=data, auth=self.auth
        )
        with logger.catch():
            r.raise_for_status()
        return r.json()

    def unique(self, fields: List[str], **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {"fields": fields, "unique": True}
        r = self.client.post(
            self.summary_url,
            headers=self.headers,
            params=params,
            json=labels,
            auth=self.auth,
        )
        with logger.catch():
            r.raise_for_status()
        data = r.json()

        results = {field: data[field]["unique"] for field in fields}
        if len(fields) == 1:
            return results[fields[0]]

        results = from_json(results)

        return results

    def max(self, fields: List[str], **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {"fields": fields, "max": True}
        r = self.client.post(
            self.summary_url,
            headers=self.headers,
            params=params,
            json=labels,
            auth=self.auth,
        )

        with logger.catch():
            r.raise_for_status()
        data = r.json()

        results = {field: data[field]["max"] for field in fields}
        results = from_json(results)

        if len(fields) == 1:
            return results[fields[0]]
        return results

    def min(self, fields: List[str], **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {"fields": fields, "min": True}
        r = self.client.post(
            self.summary_url,
            headers=self.headers,
            params=params,
            json=labels,
            auth=self.auth,
        )

        with logger.catch():
            r.raise_for_status()
        data = r.json()

        results = {field: data[field]["min"] for field in fields}
        results = from_json(results)
        if len(fields) == 1:
            return results[fields[0]]
        return results

    def count(self, **labels):
        labels = jsonable(labels)
        params = {"count": "true"}
        r = self.client.post(
            self.summary_url,
            headers=self.headers,
            params=params,
            json=labels,
            auth=self.auth,
        )

        with logger.catch():
            r.raise_for_status()
        data = r.json()
        cnt = data.get("count", None)
        if cnt is None:
            raise ValueError("Failed to fetch count from server.")
        return int(cnt)

    def summary(self, fields: List[str] = None, **labels):
        if not isinstance(fields, list):
            fields = [fields]
        labels = jsonable(labels)
        params = {
            "fields": fields,
            "min": True,
            "max": True,
            "unique": True,
            "count": True,
        }
        r = self.client.post(
            self.summary_url,
            headers=self.headers,
            params=params,
            json=labels,
            auth=self.auth,
        )

        with logger.catch():
            r.raise_for_status()
        data = r.json()
        data = from_json(data)
        if len(fields) == 1:
            return data[fields[0]]
        return data
