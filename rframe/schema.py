import inspect
import json

import pandas as pd
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo, ModelField
from typing import Any, Dict, List, Mapping, Optional, Union, Generator


from .dispatchers import are_equal
from .indexes import BaseIndex, Index, MultiIndex
from .interfaces import get_interface
from .interfaces.pandas import to_pandas

RESERVED_FIELDS = ("limit", "skip", "sort")


class EditError(Exception):
    pass


class InsertionError(EditError):
    pass


class UpdateError(EditError):
    pass


class DeletionError(EditError):
    pass


class BaseSchema(BaseModel):
    class Config:
        validate_assignment = True

    def __init_subclass__(cls) -> None:
        for name in RESERVED_FIELDS:
            if name in cls.__fields__:
                raise ValueError(f"Field name '{name}' is reserved and cannot be used.")
        return super().__init_subclass__()

    @classmethod
    def register_datasource(cls, datasource, name='data', initialize=False):
        """Register a datasource with this schema
        """

        if hasattr(cls, name):
            raise ValueError(f"Datasource name '{name}' is already registered.")
            
        from rframe.data_accessor import DataAccessor

        accessor = DataAccessor(cls, datasource, initdb=initialize)
        setattr(cls, name, accessor)

    @classmethod
    def default_datasource(cls):
        return cls.empty_dframe()

    @classmethod
    def field_info(cls) -> Dict[str, FieldInfo]:
        return {name: field.field_info for name, field in cls.__fields__.items()}

    @classmethod
    def get_index_fields(cls) -> Dict[str, ModelField]:
        fields = {}
        for name, field in cls.__fields__.items():
            if isinstance(field.field_info, BaseIndex):
                fields[name] = field
        return fields

    @classmethod
    def get_column_fields(cls) -> Dict[str, ModelField]:
        fields = {}
        for name, field in cls.__fields__.items():
            if not isinstance(field.field_info, BaseIndex):
                fields[name] = field
        return fields

    @classmethod
    def get_query_signature(cls, default=None):
        params = []
        for name, field in cls.__fields__.items():
            alias = field.alias
            for type_ in cls.mro():
                if name in getattr(type_, "__annotations__", {}):
                    label_annotation = type_.__annotations__[name]
                    annotation = Optional[
                        Union[
                            label_annotation,
                            List[label_annotation],
                            Dict[str, Optional[label_annotation]],
                        ]
                    ]
                    param = inspect.Parameter(
                        name,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default,
                        annotation=annotation,
                        )
                    params.append(param)

                    if name == alias:
                        break

                    alias_param = param = inspect.Parameter(
                        alias,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default,
                        annotation=annotation,
                        )
                    params.append(alias_param)
                    break
                
        return inspect.Signature(params)

    @classmethod
    def get_index(cls) -> BaseIndex:
        index_fields = cls.get_index_fields()
        indexes = []
        for name, field in index_fields.items():
            index: BaseIndex = field.field_info
            index.__set_name__(cls, name)
            indexes.append(index)

        if len(indexes) == 1:
            index = indexes[0]
        else:
            index = MultiIndex(*indexes)

        return index

    @classmethod
    def index_for(cls, name):
        """Fetches index instance for the given field
        If field_info is index type, returns it
        otherwise return a simple Index instance.
        This allows for queries on non-index fields.
        """
        if isinstance(name, list):
            return MultiIndex(*[cls.index_for(n) for n in name])

        if name not in cls.__fields__:
            raise KeyError(f"{name} is not a valid" f"field for schema {cls.__name__}")

        field_info = cls.field_info().get(name, None)
        if not isinstance(field_info, BaseIndex):
            field_info = Index()
        field_info.__set_name__(cls, name)
        return field_info

    @classmethod
    def validate_partial(cls, allow_None=True, **kwargs):
        """Perform validation on subset of fields"""
        validated = {}
        errors = []
        for name, field in cls.__fields__.items():
            if name not in kwargs:
                continue
            val = kwargs[name]
            if val is None and allow_None:
                validated[name] = val
                continue
            val, error = field.validate(val, validated, loc=name)
            if errors:
                errors.append(error)

            validated[name] = val

        if errors:
            raise ValidationError(errors)

        if len(validated) == 1:
            return validated[list(validated)[0]]

        return validated

    @classmethod
    def rframe(cls, datasource=None):
        """Contruct a RemoteFrame from this schema and
        datasource.
        """
        import rframe

        if datasource is None:
            datasource = cls.default_datasource()
        return rframe.RemoteFrame(cls, datasource)

    @classmethod
    def extract_labels(cls, **kwargs):
        """Extract query labels from kwargs

        returns extracted labels and remaining kwargs
        """
        labels = {}

        for name, field in cls.__fields__.items():
            label = kwargs.pop(name, None)
            if label is None:
                label = kwargs.pop(field.alias, None)
            if label is None:
                continue
            labels[name] = label

        return labels, kwargs

    @classmethod
    def compile_query(cls, datasource=None, **labels):
        if datasource is None:
            datasource = cls.default_datasource()

        labels, kwargs = cls.extract_labels(**labels)
        for name in cls.get_index_fields():
            if name not in labels:
                labels[name] = None

        indexes = [cls.index_for(name) for name in labels]
        if len(indexes) == 1:
            index = indexes[0]
            label = labels[index.name]
        else:
            index = MultiIndex(*indexes)
            label = labels

        label = index.validate_label(label)

        kwargs = {k.lstrip("_"): v for k, v in kwargs.items()}
        interface = get_interface(datasource, **kwargs)

        query = interface.compile_query(index, label)
        return query

    @classmethod
    def _find(
        cls, datasource=None, skip=None, limit=None, sort=None, **labels
    ) -> Generator[dict, None, None]:
        """Internal find function, performs data validation but
        returns raw dicts, not schema instances.
        """

        query = cls.compile_query(datasource=datasource, **labels)
        for doc in query.iter(limit=limit, skip=skip, sort=sort):
            try:
                cls.validate(doc)
            except ValidationError:
                continue
            yield doc

    @classmethod
    def find(
        cls, datasource=None, skip=None, limit=None, sort=None, **labels
    ) -> List["BaseSchema"]:
        """Find documents in datasource matching the given labels
        returns List[BaseSchema]
        """

        return [
            cls(**doc)
            for doc in cls._find(
                datasource, skip=skip, limit=limit, sort=sort, **labels
            )
        ]

    @classmethod
    def find_iter(cls, datasource=None, skip=None, limit=None, sort=None, **labels):
        for doc in cls._find(datasource, skip=skip, limit=limit, sort=sort, **labels):
            yield cls(**doc)

    @classmethod
    def find_df(
        cls, datasource=None, skip=None, limit=None, sort=None, **labels
    ) -> pd.DateOffset:
        docs = [
            to_pandas(d)
            for d in cls._find(datasource, skip=skip, limit=limit, sort=sort, **labels)
        ]
        df = pd.json_normalize(docs)
        if not len(df):
            df = df.reindex(columns=list(cls.__fields__))
        index_fields = list(cls.get_index_fields())
        if len(index_fields) == 1:
            index_fields = index_fields[0]
        return df.set_index(index_fields)

    @classmethod
    def find_one(
        cls, datasource=None, skip=None, sort=None, **labels
    ) -> Optional["BaseSchema"]:
        docs = cls.find(datasource=datasource, skip=skip, limit=1, sort=sort, **labels)
        if docs:
            return docs[0]
        return None

    @classmethod
    def from_pandas(cls, record):
        if isinstance(record, list):
            return [cls.from_pandas(d) for d in record]
        if isinstance(record, pd.DataFrame):
            return [cls.from_pandas(d) for d in record.to_dict(orient="records")]

        if not isinstance(record, Mapping):
            raise TypeError(
                "Record must be of type Mapping,"
                "List[Mapping] or DataFrame],"
                f"got {type(record)}"
            )

        data = dict(record)
        for name in cls.get_index_fields():
            index = cls.index_for(name)
            label = record.get(name, None)
            data[name] = index.from_pandas(label)
        return cls(**data)

    @classmethod
    def initdb(cls, datasource, **kwargs):
        interface = get_interface(datasource, **kwargs)
        interface.initdb(cls)

    @classmethod
    def unique(cls, datasource=None, fields: Union[str, List[str]] = None, **labels):
        if fields is None:
            fields = list(cls.get_column_fields())
        elif isinstance(fields, str):
            fields = [fields]
        query = cls.compile_query(datasource, **labels)

        unique = query.unique(fields)
        if isinstance(unique, dict):
            for k, vs in unique.items():
                unique[k] = [cls.validate_partial(**{k: v}) for v in vs]
        else:
            unique = [cls.validate_partial(**{fields[0]: v}) for v in unique]
        return unique

    @classmethod
    def min(cls, datasource=None, fields: Union[str, List[str]] = None, **labels):
        if fields is None:
            fields = list(cls.get_column_fields())
        elif isinstance(fields, str):
            fields = [fields]
        query = cls.compile_query(datasource, **labels)
        min = query.min(fields)
        if isinstance(min, dict):
            for k, v in min.items():
                min[k] = cls.validate_partial(**{k: v})
        else:
            min = cls.validate_partial(**{fields[0]: min})
        return min

    @classmethod
    def max(cls, datasource=None, fields: Union[str, List[str]] = None, **labels):
        if fields is None:
            fields = list(cls.get_column_fields())
        elif isinstance(fields, str):
            fields = [fields]
        query = cls.compile_query(datasource, **labels)
        max = query.max(fields)
        if isinstance(max, dict):
            for k, v in max.items():
                max[k] = cls.validate_partial(**{k: v})
        else:
            max = cls.validate_partial(**{fields[0]: max})
        return max

    @classmethod
    def count(cls, datasource=None, **labels):
        query = cls.compile_query(datasource, **labels)
        return int(query.count())

    @property
    def index_labels(self):
        return {k: getattr(self, k) for k in self.get_index_fields()}

    @property
    def index_labels_tuple(self):
        return tuple(v for v in self.index_labels.values())

    @property
    def column_values(self):
        values = self.dict()
        return {attr: values[attr] for attr in self.get_column_fields()}

    def save(self, datasource=None, **kwargs):
        if datasource is None:
            datasource = self.default_datasource()
        interface = get_interface(datasource, **kwargs)
        existing = self.find(datasource, **self.index_labels)
        if not existing:
            # No documents found, insert new
            try:
                self.__pre_insert(datasource)
                interface.insert(self)
            except Exception as e:
                self.__post_insert(datasource, exception=e)
                raise e
            self.__post_insert(datasource)

        elif len(existing) == 1:
            # Single document found, update
            try:
                existing[0].__pre_update(datasource, self)
                interface.update(self)
            except Exception as e:
                existing[0].__post_update(datasource, self, exception=e)
                raise e
            existing[0].__post_update(datasource, self)
        else:
            # Multiple documents found, raise exception
            raise UpdateError(
                "Multiple documents match document "
                f"index ({self.index_labels}). "
                "Multiple update is not supported."
            )

    def delete(self, datasource=None, **kwargs):
        if datasource is None:
            datasource = self.default_datasource()
        interface = get_interface(datasource, **kwargs)
        try:
            self.__pre_delete(datasource)
            interface.delete(self)
        except Exception as e:
            self.__post_delete(datasource, exception=e)
            raise e

    def __pre_insert(self, datasource):
        """This method is called  pre insertion
        if self.save(datasource) was called and a query on datasource
        with self.index_labels did not return any documents.

        raises an InsertionError if user defined checks fail.
        """
        try:
            self.pre_insert(datasource)
        except Exception as e:
            raise InsertionError(
                f"Cannot insert new document ({self})."
                f"The schema raised the following exception: {e}"
            )

    def __post_insert(self, datasource, exception=None):
        """This method is called post insertion
        runs the schemas post insertion hook and returns
        """
        self.post_insert(datasource, exception)

    def __pre_update(self, datasource, new):
        """This method is called if new.save(datasource)
        was called and a query on datasource
        with new.index_labels returned this document.

        raises an UpdateError if user defined checks fail.
        """
        try:
            self.pre_update(datasource, new=new)
        except Exception as e:
            raise UpdateError(
                f"Cannot update existing instance ({self}) "
                f"with new instance ({new}), the schema "
                f"raised the following exception: {e}"
            )

    def __post_update(self, datasource, new, exception=None):
        """This method is called after updates
        runs the schemas post update hook and returns
        """
        self.post_update(datasource, new, exception)

    def __pre_delete(self, datasource):
        """This method is called pre deletion
        if self.delete(datasource) was called and a query on datasource
        with self.index_labels returned this document.

        raises an DeletionError if user defined checks fail.
        """
        try:
            self.pre_delete(datasource)
        except Exception as e:
            raise DeletionError(
                f"Cannot delete document ({self})."
                f"The schema raised the following exception: {e}"
            )

    def __post_delete(self, datasource, exception=None):
        """This method is called post deletion
        runs the schemas post deletion hook and returns
        """
        self.post_delete(datasource, exception)

    def pre_insert(self, datasource):
        """Pre insert hook for user
        defined checks to perform
        prior to document insertion.
        Should raise an exception if insertion
        is disallowed.
        """
        pass

    def post_insert(self, datasource, exception=None):
        """User defined hook to perform
        after document insertion.
        """
        pass

    def pre_update(self, datasource, new):
        """User defined checks to perform
        prior to document update.
        Should raise an exception if update
        is disallowed.
        """
        pass

    def post_update(self, datasource, new, exception=None):
        """User defined hook to perform
        after document updates.
        """
        pass

    def pre_delete(self, datasource):
        """User defined checks to perform
        prior to document deletion.
        Should raise an exception if deletion
        is disallowed.
        """
        pass

    def post_delete(self, datasource, exception=None):
        """User defined hook to perform
        after document deletion.
        """
        pass

    def same_values(self, other):
        if other is None:
            return False
        if not isinstance(other, BaseSchema):
            return False
        return are_equal(self.column_values, other.column_values)

    def same_index(self, other):
        if other is None:
            return False
        if not isinstance(other, BaseSchema):
            return False
        return are_equal(self.index_labels, other.index_labels)

    def pandas_dict(self):
        return to_pandas(self.dict())

    @classmethod
    def empty_dframe(cls):
        columns = list(cls.__fields__)
        indexes = list(cls.get_index_fields())
        if len(indexes) == 1:
            indexes = indexes[0]
        return pd.DataFrame().reindex(columns=columns).set_index(indexes)

    def dframe(self):
        index_fields = list(self.get_index_fields())
        if len(index_fields) == 1:
            index_fields = index_fields[0]
        df = pd.DataFrame([self.pandas_dict()])
        return df.set_index(index_fields)

    def jsonable(self):
        return json.loads(self.json())

    @property
    def raw_index_labels(self):
        return tuple(getattr(self, k) for k in self.get_index_fields())

    def __lt__(self, other: "BaseSchema"):
        return self.raw_index_labels < other.raw_index_labels

    def __le__(self, other: "BaseSchema"):
        if not isinstance(other, BaseSchema):
            raise TypeError("")
        return self.raw_index_labels <= other.raw_index_labels

    def __eq__(self, other: Any):
        if not isinstance(other, BaseSchema):
            False
        return are_equal(self.dict(), other.dict())

    def __gt__(self, other: "BaseSchema"):
        return self.raw_index_labels > other.raw_index_labels

    def __ge__(self, other: "BaseSchema"):
        return self.raw_index_labels >= other.raw_index_labels
