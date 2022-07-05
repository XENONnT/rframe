"""Top-level package for rframe."""

__author__ = """Yossi Mosbacher"""
__email__ = "joe.mosbacher@gmail.com"
__version__ = "0.1.18"

from loguru import logger

from . import schema, indexes, utils, dispatchers

from .indexes import Index, InterpolatingIndex, IntervalIndex
from .types import Interval, IntegerInterval, TimeInterval
from .rframe import RemoteFrame
from .schema import BaseSchema
from .rest_client import BaseRestClient, RestClient
from .utils import jsonable

try:
    from .rest_server import SchemaRouter
except ImportError:
    pass

logger.disable("rframe")
