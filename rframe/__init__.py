"""Top-level package for rframe."""

__author__ = """Yossi Mosbacher"""
__email__ = "joe.mosbacher@gmail.com"
__version__ = "0.1.10"

from .indexes import *
from .rframe import RemoteFrame
from .schema import BaseSchema
from .rest_client import BaseRestClient, RestClient
from .rest_server import SchemaRouter
from .utils import jsonable
from loguru import logger

logger.disable("rframe")
