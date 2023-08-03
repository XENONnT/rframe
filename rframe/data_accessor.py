import inspect
import makefun
import toolz
import pandas as pd

from pydantic import ValidationError
from typing import Any, Generator, Iterable, Optional

from rframe.schema import DeletionError, InsertionError, UpdateError

from .interfaces.pandas import to_pandas


class DataAccessor:
    schema: "BaseSchema"
    storage: Any
    initialized: bool = False

    @property
    def index_names(self):
        return tuple(self.schema.get_index_fields())

    @property
    def column_names(self):
        return tuple(self.schema.get_column_fields())

    @property
    def rframe(self):
        return self.schema.rframe(self.storage)

    def __init__(self, schema, datasource, initdb=False):
        self.schema = schema
        self.storage = datasource


        for name in dir(self):

            if name.startswith("__"):
                continue

            if not name.startswith("_"):
                continue

            impl = getattr(self, name)
            query_parameters = list(
                self.schema.get_query_signature().parameters.values()
            )
            func_signature = inspect.signature(impl)
            signature = makefun.add_signature_parameters(
                func_signature, first=query_parameters
            )
            method = makefun.create_function(signature, impl, func_name=name)
            setattr(self, name[1:], method)

        if initdb:
            self.initdb()

    def _find(
        self, skip=None, limit=None, sort=None, **labels
    ) -> Generator[dict, None, None]:
        """Internal find function, performs data validation but
        returns raw dicts, not schema instances.
        """
        labels = {k: v for k, v in labels.items() if v is not None}
        query = self.schema.compile_query(datasource=self.storage, **labels)
        returned = set()
        for doc in query.iter(limit=limit, skip=skip, sort=sort):
            try:
                doc = self.schema(**doc)
                if doc.index_labels_tuple in returned:
                    continue
                returned.add(doc.index_labels_tuple)
                doc = doc.dict()
            except ValidationError:
                continue
            yield doc

    def _find_dicts(self, skip=None, limit=None, sort=None, **labels):
        return list(self._find(skip=skip, limit=limit, sort=sort, **labels))

    def _find_docs(self, skip=None, limit=None, sort=None, **labels):
        return list(self._find_iter(skip=skip, limit=limit, sort=sort, **labels))

    def _find_iter(self, skip=None, limit=None, sort=None, **labels):
        for doc in self._find(skip=skip, limit=limit, sort=sort, **labels):
            yield self.schema(**doc)

    def _find_df(self, skip=None, limit=None, sort=None, **labels) -> pd.DataFrame:
        docs = [
            to_pandas(d)
            for d in self._find_dicts(skip=skip, limit=limit, sort=sort, **labels)
        ]
        df = pd.json_normalize(docs)
        if not len(df):
            df = df.reindex(columns=list(self.schema.__fields__))
        index_fields = list(self.schema.get_index_fields())
        if len(index_fields) == 1:
            index_fields = index_fields[0]
        return df.set_index(index_fields)

    def _find_one(self, skip=None, sort=None, **labels) -> Optional["BaseSchema"]:
        docs = self._find_docs(skip=skip, limit=1, sort=sort, **labels)
        if docs:
            return docs[0]
        return None

    def _min(self, fields=None, **labels) -> Any:
        if fields is None:
            fields = list(self.schema.__fields__)
        elif isinstance(fields, str):
            fields = [fields]

        labels = {k: v for k, v in labels.items() if v is not None}
        query = self.schema.compile_query(self.storage, **labels)

        result = query.min(fields)
        if isinstance(result, dict):
            result = { k: self.schema.validate_partial(**{k: v})
                        for k,v in result.items() }
        else:
            result = self.schema.validate_partial(**{fields[0]: result})
        return result

    def _max(self, fields=None, **labels) -> Any:
        if fields is None:
            fields = list(self.schema.__fields__)
        elif isinstance(fields, str):
            fields = [fields]

        labels = {k: v for k, v in labels.items() if v is not None}
        query = self.schema.compile_query(self.storage, **labels)

        result = query.max(fields)
        if isinstance(result, dict):
            result = { k: self.schema.validate_partial(**{k: v})
                        for k,v in result.items() }
        else:
            result = self.schema.validate_partial(**{fields[0]: result})
        return result

    def _unique(self, fields=None, **labels) -> Any:
        if fields is None:
            fields = list(self.schema.__fields__)
        elif isinstance(fields, str):
            fields = [fields]

        labels = {k: v for k, v in labels.items() if v is not None}
        query = self.schema.compile_query(self.storage, **labels)

        result = query.unique(fields)
        if isinstance(result, dict):
            result = { k: [self.schema.validate_partial(**{k: v}) for v in vs]
                        for k, vs in result.items() }
        else:
            result = [self.schema.validate_partial(**{fields[0]: v}) for v in result]
        return result

    def _count(self, **labels):
        labels = {k: v for k, v in labels.items() if v is not None}
        query = self.schema.compile_query(self.storage, **labels)
        return int(query.count())

    def insert(self, docs, raise_on_error=True, dry=False):
        if not self.initialized:
            self.initdb()

        if not isinstance(docs, (list, tuple)):
            docs = [docs]

        res = {
            "success": [],
            "failed": [],
            "errors": [],
        }
        for doc in docs:
            if not isinstance(doc, self.schema):
                doc = self.schema(**doc)
            try:
                doc.save(self.storage, dry=dry)
                res["success"].append(doc)
            except (InsertionError, UpdateError) as e:
                if raise_on_error:
                    raise e
                res["failed"].append(doc)
                res["errors"].append(e)
        if raise_on_error:
            return res['success']
        return res

    def delete(self, docs, raise_on_error=True):
        if not isinstance(docs, (list, tuple)):
            docs = [docs]

        res= {
            "success": [],
            "failed": [],
            "errors": [],
            }

        for doc in docs:
            if not isinstance(doc, self.schema):
                doc = self.schema(**doc)

            try:
                doc.delete(self.storage)
                res["success"].append(doc)
            except DeletionError as e:
                if raise_on_error:
                    raise e

        if raise_on_error:
            return res["success"]
        return res

    def initdb(self):
        self.schema.initdb(self.storage)
        self.initialized = True
