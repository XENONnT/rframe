import re
import json
import math
import inspect
from typing import overload
import jsonschema

from numbers import Integral, Number, Real
from functools import partial, singledispatch
from pydantic import BaseModel
from pydantic.json import ENCODERS_BY_TYPE
from collections.abc import Iterable, Mapping

from .types import Interval


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def snake_to_camel(name):
    return name.title().replace("_", "")


def get_all_subclasses(type_):
    subclasses = []
    for subclass in type_.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(get_all_subclasses(subclass))
    return subclasses


def filter_kwargs(func, kwargs):
    """Filter out keyword arguments that
    are not in the call signature of func
    and return filtered kwargs dictionary
    """
    params = inspect.signature(func).parameters
    if any([str(p).startswith("**") for p in params.values()]):
        # if func accepts wildcard kwargs, return all
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def jsonable(obj):

    if obj is None:
        return obj

    if isinstance(obj, BaseModel):
        return json.loads(obj.json())

    if isinstance(obj, Mapping):
        return {k: jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, set, frozenset, tuple)):
        return [jsonable(v) for v in obj]

    if isinstance(obj, (str, int, float, type(None))):
        return obj

    if type(obj) in ENCODERS_BY_TYPE:
        return ENCODERS_BY_TYPE[type(obj)](obj)

    if isinstance(obj, Real):
        return float(obj)

    if isinstance(obj, Integral):
        return int(obj)

    if isinstance(obj, slice):
        return {
            "start": jsonable(obj.start),
            "stop": jsonable(obj.stop),
            "step": jsonable(obj.step),
        }

    raise TypeError(f"Cannot convert {type(obj)} to JSON")


def as_bson_schema(schema, resolver=None):

    if not isinstance(schema, dict):
        return schema

    if resolver is None:
        resolver = jsonschema.RefResolver.from_schema(schema)

    if "$ref" in schema:
        return as_bson_schema(resolver.resolve(schema["$ref"])[1], resolver=resolver)
    new = {}

    for k, v in schema.items():
        if k in ["definitions", "format", "default"]:
            continue
        if k == "type":
            k = "bsonType"
        if v == "integer":
            v = "int"
        elif v == "boolean":
            v = "bool"
        elif v == "string" and schema.get("format", "") == "date-time":
            v = "date"
        elif isinstance(v, dict):
            v = as_bson_schema(v, resolver=resolver)
        elif isinstance(v, list):
            v = [as_bson_schema(vi, resolver=resolver) for vi in v]
        new[k] = v
    return new


"""
Copied from python 3.8 functools for 3.7 support
"""


WRAPPER_ASSIGNMENTS = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
)
WRAPPER_UPDATES = ("__dict__",)


def update_wrapper(
    wrapper, wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES
):
    """Update a wrapper function to look like the wrapped function
    wrapper is the function to be updated
    wrapped is the original function
    assigned is a tuple naming the attributes assigned directly
    from the wrapped function to the wrapper function (defaults to
    functools.WRAPPER_ASSIGNMENTS)
    updated is a tuple naming the attributes of the wrapper that
    are updated with the corresponding attribute from the wrapped
    function (defaults to functools.WRAPPER_UPDATES)
    """
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


# Descriptor version
class singledispatchmethod:
    """Single-dispatch generic method descriptor.
    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.
    """

    def __init__(self, func):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError(f"{func!r} is not callable or a descriptor")

        self.dispatcher = singledispatch(func)
        self.func = func

    def register(self, cls, method=None):
        """generic_method.register(cls, func) -> func
        Registers a new implementation for the given *cls* on a *generic_method*.
        """
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls=None):
        def _method(*args, **kwargs):
            class_ = args[0] if isinstance(args[0], type) else args[0].__class__
            method = self.dispatcher.dispatch(class_)
            return method.__get__(obj, cls)(*args, **kwargs)

        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method.register = self.register
        update_wrapper(_method, self.func)
        return _method

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, "__isabstractmethod__", False)


class FrozenDict(Mapping):
    """https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be"""

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash

    def __lt__(self, other):
        if isinstance(other, Mapping) and self.keys() == other.keys():
            return tuple(self.values()) < tuple(other.values())
        raise TypeError(
            f"< not supported between instances of {type(self)} and {type(other)}"
        )

    def __gt__(self, other):
        if isinstance(other, Mapping) and self.keys() == other.keys():
            return tuple(self.values()) > tuple(other.values())
        raise TypeError(
            f"> not supported between instances of {type(self)} and {type(other)}"
        )

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other


def hashable_doc(doc):
    if isinstance(doc, dict):
        doc = {k: hashable_doc(v) for k, v in doc.items()}
        return FrozenDict(doc)
    return doc


def unhashable_doc(doc):
    if isinstance(doc, FrozenDict):
        doc = {k: unhashable_doc(v) for k, v in doc.items()}
        return dict(doc)
    return doc
