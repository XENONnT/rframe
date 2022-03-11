
from ast import Import
import makefun

from typing import (Any, Callable, Dict, Generic,
                    List, Optional, Sequence, Type, Union)

from .schema import BaseSchema

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.types import DecoratedCallable
    from fastapi.params import Depends


    NOT_FOUND = HTTPException(404, "Item not found")
    DEPENDENCIES = Optional[Sequence[Depends]]


    class SchemaRouter(APIRouter):
        schema: Type[BaseSchema]
        _base_path: str = "/"

        def __init__(
            self,
            schema: Type[BaseSchema],
            datasource = None,
            prefix: Optional[str] = None,
            tags: Optional[List[str]] = None,
            # paginate: Optional[int] = None,
            query_route: bool = True,
            insert_route: bool = True,
            **kwargs: Any,
        ) -> None:

            self.schema = schema
            self.datasource = datasource

            prefix = str(prefix if prefix else self.schema.__name__).lower()
            prefix = self._base_path + prefix.strip("/")
            tags = tags or [prefix.strip("/").capitalize()]

            super().__init__(prefix=prefix, tags=tags, **kwargs)

            if query_route:
                
                self._add_api_route(
                    "",
                    self._query_route(),
                    methods=["GET"],
                    response_model = Optional[List[self.schema]],  # type: ignore
                    summary=f"Query {self.schema.__name__} documents",
                )

            if insert_route:
                self._add_api_route(
                    "",
                    self._insert_route(),
                    methods=["POST"],
                    response_model=Optional[self.schema],  # type: ignore
                    summary=f"Insert One {self.schema.__name__} document",
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

        def _query_route(self,*args, **kwargs) -> Callable[..., Any]:
            query_func_name = f"{self.schema.__name__}_query"
            query_signature = self.schema.get_query_signature()

            find_with_source = makefun.partial(self.schema.find,
                                            datasource=self.datasource)

            query_func = makefun.create_function(query_signature,
                                                find_with_source,
                                                func_name=query_func_name)

            return query_func

        def _insert_route(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
            async def insert(doc: self.schema):
                return doc.save(self.datasource)
            return insert

except ImportError:
    class SchemaRouter:
        def __init__(self) -> None:
            raise ImportError("fastapi is not installed")