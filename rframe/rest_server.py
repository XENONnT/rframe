from enum import Enum
import inspect
import makefun

from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from rframe.interfaces.json import from_json

from .schema import BaseSchema
from .utils import camel_to_snake

from fastapi import APIRouter, HTTPException, Query
from fastapi.types import DecoratedCallable
from fastapi.params import Depends

NOT_FOUND = HTTPException(404, "Item not found")
DEPENDENCIES = Optional[Sequence[Depends]]
Model = Mapping[Any, Any]


class SchemaRouter(APIRouter):
    schema: Type[BaseSchema]
    _base_path: str = "/"

    def __init__(
        self,
        schema: Type[BaseSchema],
        datasource=None,
        prefix: Optional[str] = None,
        query_path="/query",
        insert_path="/insert",
        update_path="/update",
        delete_path="/delete",
        summary_path="/summary",
        tags: Optional[List[Union[str, Enum]]] = None,
        can_read: Union[bool, DEPENDENCIES] = True,
        can_write: Union[bool, DEPENDENCIES] = True,
        **kwargs: Any,
    ) -> None:

        self.schema = schema
        self.datasource = datasource

        prefix = prefix if prefix else camel_to_snake(self.schema.__name__)
        prefix = self._base_path + prefix.strip("/")
        tags = tags or [prefix.strip("/").capitalize()]

        super().__init__(prefix=prefix, tags=tags, **kwargs)

        if isinstance(can_read, Depends):
            can_read = [can_read]
        if can_read:

            self._add_api_route(
                "/",
                self._schema_route(),
                methods=["GET"],
                summary=f"Get json schema for {self.schema.__name__} document",
                dependencies=can_read,
            )

            self._add_api_route(
                query_path,
                self._query_route(),
                methods=["POST"],
                response_model=Optional[List[self.schema]],  # type: ignore
                summary=f"Perform query on {self.schema.__name__} documents",
                dependencies=can_read,
            )

            self._add_api_route(
                summary_path,
                self._summary_route(),
                methods=["POST"],
                response_model=Optional[dict],  # type: ignore
                summary=f"Query summary info on {self.schema.__name__} documents",
                dependencies=can_read,
            )

        if isinstance(can_write, Depends):
            can_write = [can_write]

        if can_write:
            self._add_api_route(
                insert_path,
                self._insert_one_route(),
                methods=["POST"],
                response_model=Optional[self.schema],  # type: ignore
                summary=f"Insert One {self.schema.__name__} document",
                dependencies=can_write,
            )

            self._add_api_route(
                insert_path,
                self._insert_many_route(),
                methods=["POST"],
                response_model=List[Union[self.schema, str]],  # type: ignore
                summary=f"Insert multiple {self.schema.__name__} documents",
                dependencies=can_write,
            )

            self._add_api_route(
                update_path,
                self._update_one_route(),
                methods=["PUT"],
                response_model=Optional[self.schema],  # type: ignore
                summary=f"Insert One {self.schema.__name__} document",
                dependencies=can_write,
            )

            self._add_api_route(
                delete_path,
                self._delete_route(),
                methods=["DELETE"],
                response_model=Optional[self.schema],  # type: ignore
                summary=f"Delete one {self.schema.__name__} document",
                dependencies=can_write,
            )

    def _add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        dependencies: Union[bool, DEPENDENCIES],
        error_responses: Optional[List[HTTPException]] = None,
        **kwargs: Any,
    ) -> None:
        dependencies = [] if isinstance(dependencies, bool) else dependencies
        responses: Any = (
            {err.status_code: {"detail": err.detail} for err in error_responses}
            if error_responses
            else None
        )

        super().add_api_route(
            path, endpoint, dependencies=dependencies, responses=responses, **kwargs
        )

    def api_route(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        """Overrides and exiting route if it exists"""
        methods = kwargs["methods"] if "methods" in kwargs else ["GET"]
        self.remove_api_route(path, methods)
        return super().api_route(path, *args, **kwargs)

    def get(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        self.remove_api_route(path, ["Get"])
        return super().get(path, *args, **kwargs)

    def post(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        self.remove_api_route(path, ["POST"])
        return super().post(path, *args, **kwargs)

    def put(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        self.remove_api_route(path, ["PUT"])
        return super().put(path, *args, **kwargs)

    def delete(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        self.remove_api_route(path, ["DELETE"])
        return super().delete(path, *args, **kwargs)

    def remove_api_route(self, path: str, methods: List[str]) -> None:
        methods_ = set(methods)

        for route in self.routes:
            if (
                route.path == f"{self.prefix}{path}"  # type: ignore
                and route.methods == methods_  # type: ignore
            ):
                self.routes.remove(route)

    def _collect_stats(self, query, field, min, max, unique, **kwargs):

        results = {}
        if min:
            results["min"] = query.min(field)
        if max:
            results["max"] = query.max(field)
        if unique:
            results["unique"] = query.unique(field)

        return results

    def _summary_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:

        # This is the actual implementation that will be called when
        # a request is made to the route.
        def summary_func_impl(
            fields: List[str] = None,
            unique: bool = False,
            min: bool = False,
            max: bool = False,
            count: bool = False,
            **kwargs
        ) -> dict:
            query = self.schema.compile_query(self.datasource, **kwargs)
            if fields is None:
                fields = []
            results = {}
            for field in fields:
                results[field] = self._collect_stats(
                    query, field, min, max, unique, **kwargs
                )
            if count:
                results["count"] = query.count()
            return results
        # fastapi uses the function signature to generate the openapi docs
        # and request body validation. We need to edit the signature to include
        # the query signature from the schema instead of the generic **kwargs.
        summary_func_name = f"{self.schema.__name__}_summary"
        query_signature = self.schema.get_query_signature(default=None)

        # Also add the extra query parameters for summary customization.
        # set all defaults to False, since queries may be expensive.
        extra = [
            inspect.Parameter(
                "fields",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Query(None),
                annotation=List[str],
            ),
            inspect.Parameter(
                "unique",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Query(False),
                annotation=bool,
            ),
            inspect.Parameter(
                "min",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Query(False),
                annotation=bool,
            ),
            inspect.Parameter(
                "max",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Query(False),
                annotation=bool,
            ),
            inspect.Parameter(
                "count",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Query(False),
                annotation=bool,
            ),
        ]

        query_signature = makefun.add_signature_parameters(query_signature, extra)
        summary_func = makefun.create_function(
            query_signature, summary_func_impl, func_name=summary_func_name
        )
        return summary_func

    def _query_route(self, *args, **kwargs) -> Callable[..., Any]:
        # This is the actual implementation that will be called when
        # a request is made to the route.
        def query_func(limit: int = None, skip: int = None, sort=None, kwargs: dict = {}):
            kwargs = {k: from_json(v) for k, v in kwargs.items() if v is not None}
            query = self.schema.compile_query(self.datasource, **kwargs)
            return [
                self.schema(**d)
                for d in query.execute(limit=limit, skip=skip, sort=sort)
            ]
        return query_func
    
        # def query_func_impl(
        #     limit: int = None, skip: int = None, sort=None, **kwargs
        # ) -> List[BaseSchema]:
        #     kwargs = {k: from_json(v) for k, v in kwargs.items() if v is not None}
        #     query = self.schema.compile_query(self.datasource, **kwargs)
        #     return [
        #         self.schema(**d)
        #         for d in query.execute(limit=limit, skip=skip, sort=sort)
        #     ]

        # # fastapi uses the function signature to generate the openapi docs
        # # and request body validation. We need to edit the signature to include
        # # the query signature from the schema instead of the generic **kwargs.
        # query_func_name = f"{self.schema.__name__}_query"
        # query_signature = self.schema.get_query_signature(default=None)

        # # Also add the extra query parameters for pagination.
        # extra = [
        #     inspect.Parameter(
        #         "limit",
        #         inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #         default=Query(None),
        #         annotation=int,
        #     ),
        #     inspect.Parameter(
        #         "skip",
        #         inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #         default=Query(None),
        #         annotation=int,
        #     ),
        #     inspect.Parameter(
        #         "sort",
        #         inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #         default=Query(None),
        #         annotation=list,
        #     ),
        # ]
        # query_signature = makefun.add_signature_parameters(query_signature, extra)

        # # Create the function with the correct signature.
        # query_func = makefun.create_function(
        #     query_signature, query_func_impl, func_name=query_func_name
        # )
        # return query_func

    def _insert_one_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        def insert_func(doc: dict) -> dict:
            doc = self.schema(**doc)
            try:
                doc.save(self.datasource)
                return doc.dict()
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        return insert_func
    
        # def insert(doc: BaseSchema) -> BaseSchema:
        #     try:
        #         doc.save(self.datasource)
        #         return doc
        #     except Exception as e:
        #         raise HTTPException(status_code=400, detail=str(e))

        # parameter = inspect.Parameter(
        #     "doc",
        #     inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #     annotation=self.schema,
        # )
        # signature = inspect.Signature(
        #     parameters=[parameter], return_annotation=self.schema
        # )
        # return makefun.create_function(
        #     signature, insert, func_name=f"{self.schema.__name__}_insert_one"
        # )

    def _update_one_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        def update_func(doc: dict) -> dict:
            doc = self.schema(**doc)
            try:
                doc.save(self.datasource)
                return doc.dict()
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        return update_func
    
        # def update(doc: BaseSchema) -> BaseSchema:
        #     try:
        #         doc.save(self.datasource)
        #         return doc
        #     except Exception as e:
        #         raise HTTPException(status_code=400, detail=str(e))

        # parameter = inspect.Parameter(
        #     "doc",
        #     inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #     annotation=self.schema,
        # )
        # signature = inspect.Signature(
        #     parameters=[parameter], return_annotation=self.schema
        # )
        # return makefun.create_function(
        #     signature, update, func_name=f"{self.schema.__name__}_update_one"
        # )

    def _insert_many_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        def insert_many_func(docs: List[dict]) -> List[dict]:
            results = []
            for d in docs:
                doc = self.schema(**d)
                try:
                    doc.save(self.datasource)
                    results.append(doc)
                except Exception as e:
                    res = {"Error": str(e)}
                    results.append(res)
            return results
        return insert_many_func
    
        # def insert_many(docs: List[BaseSchema]) -> BaseSchema:
        #     results = []
        #     for doc in docs:
        #         try:
        #             doc.save(self.datasource)
        #             results.append(doc)
        #         except Exception as e:
        #             results.append(str(e))
        #     return results

        # parameter = inspect.Parameter(
        #     "docs",
        #     inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #     annotation=List[self.schema],
        # )
        # signature = inspect.Signature(
        #     parameters=[parameter], return_annotation=List[Union[self.schema, str]]
        # )
        # return makefun.create_function(
        #     signature, insert_many, func_name=f"{self.schema.__name__}_insert_many"
        # )

    def _delete_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        def delete_func(doc: dict) -> dict:
            doc = self.schema(**doc)
            try:
                doc.delete(self.datasource)
                return doc
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        return delete_func
    
        # def delete(doc: BaseSchema) -> BaseSchema:
        #     try:
        #         doc.delete(self.datasource)
        #         return doc
        #     except Exception as e:
        #         raise HTTPException(status_code=400, detail=str(e))

        # parameter = inspect.Parameter(
        #     "doc",
        #     inspect.Parameter.POSITIONAL_OR_KEYWORD,
        #     annotation=self.schema,
        # )
        # signature = inspect.Signature(
        #     parameters=[parameter], return_annotation=self.schema
        # )
        # return makefun.create_function(
        #     signature, delete, func_name=f"{self.schema.__name__}_delete"
        # )

    def _schema_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        async def get_schema():
            return self.schema.schema()

        return get_schema
