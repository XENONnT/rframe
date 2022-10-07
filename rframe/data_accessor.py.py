
import inspect
import makefun
import pandas as pd

from pydanitc import ValidationError
from typing import Any, Generator, Optional, Union, List

from . import BaseSchema
from .interfaces.pandas import to_pandas


METHODS = ['find', 'find_dicts', 'find_docs', 'find_df', 'find_iter', ]

extra_params = [
            inspect.Parameter(
                "skip",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[int],
            ),
            inspect.Parameter(
                "limit",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[int],
            ),
            inspect.Parameter(
                "sort",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[Union[str,List[str]]],
            ),
        ]

class DataAccessor:
    schema: BaseSchema
    storage: Any
    
    
    def __init__(self, schema, datasource):
        self.schema = schema
        self.storage = datasource
        self._init()
        
    def _init(self):
        for name in METHODS:
            method = self._gen_method(name)
            setattr(self, name, method)

    def _gen_method(self, name):
        signature = self.schema.get_query_signature()
        signature = makefun.add_signature_parameters(signature, extra_params)
        impl = getattr(self, '_' + name)
        return makefun.create_function(signature, impl, func_name=name)
    
    def _find(
        self, skip=None, limit=None, sort=None, **labels
    ) -> Generator[dict, None, None]:
        """Internal find function, performs data validation but
        returns raw dicts, not schema instances.
        """
        query = self.schema.compile_query(datasource=self.storage, **labels)
        for doc in query.iter(limit=limit, skip=skip, sort=sort):
            try:
                self.schema.validate(doc)
            except ValidationError:
                continue
            yield doc
            
    def _find_dicts(self, skip=None, limit=None, sort=None, **labels):
        return list(self.find(skip=skip, limit=limit, sort=sort, **labels))
    
    def _find_docs(self,  skip=None, limit=None, sort=None, **labels):
        return list(self.find_iter(skip=skip, limit=limit, sort=sort, **labels))
        
    def _find_iter(self, datasource=None, skip=None, limit=None, sort=None, **labels):
        for doc in self._find(
            skip=skip, limit=limit, sort=sort, **labels
        ):
            yield self.schema(**doc)

    def _find_df(
        self, skip=None, limit=None, sort=None, **labels
    ) -> pd.DateOffset:
        docs = [
            to_pandas(d)
            for d in self._find_dicts(
                skip=skip, limit=limit, sort=sort, **labels
            )
        ]
        df = pd.json_normalize(docs)
        if not len(df):
            df = df.reindex(columns=list(self.schema.__fields__))
        index_fields = list(self.schema.get_index_fields())
        if len(index_fields) == 1:
            index_fields = index_fields[0]
        return df.set_index(index_fields)

    def _find_one(
        self, skip=None, sort=None, **labels
    ) -> Optional[BaseSchema]:
        docs = self.find(
            skip=skip, limit=1, sort=sort, **labels
        )
        if docs:
            return docs[0]
        return None
