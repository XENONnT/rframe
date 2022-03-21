import rframe



import os
import unittest
from typing import List

import rframe
import pymongo
import tempfile

import pandas as pd

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rframe.interfaces import get_interface

from rframe.interfaces.rest import RestInterface
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

from .test_schemas import *


class TestRest(unittest.TestCase):
    """
    Test the Rest interface

    """
    def setUp(self):
        db = TinyDB(storage=MemoryStorage)

        app = FastAPI()
        self.tables = {}
        self.datasources = {}
        for name, schema in TEST_SCHEMAS.items():

            table = db.table(name)

            self.tables[schema] = table
            router = rframe.SchemaRouter(
                    schema,
                    table,
                    prefix=name,
                    )
            app.include_router(router)

        client = TestClient(app)
        for name, schema in TEST_SCHEMAS.items():
            source = rframe.RestClient('/'+name, client=client)
            self.datasources[schema] = source

    @given(SimpleSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_schema(self, docs: List[SimpleSchema]):
        self.tables[SimpleSchema].truncate()
        datasource = self.datasources[SimpleSchema]
        SimpleSchema.test(self, datasource, docs)
       
    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        self.tables[SimpleMultiIndexSchema].truncate()
        datasource = self.datasources[SimpleMultiIndexSchema]
        SimpleMultiIndexSchema.test(self, datasource, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None)
    def test_interpolated(self, docs: InterpolatingSchema):
        self.tables[InterpolatingSchema].truncate()
        datasource = self.datasources[InterpolatingSchema]
        InterpolatingSchema.test(self, datasource, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        self.tables[IntegerIntervalSchema].truncate()
        datasource = self.datasources[IntegerIntervalSchema]
        IntegerIntervalSchema.test(self, datasource, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_time_interval(self, docs: TimeIntervalSchema):
        self.tables[TimeIntervalSchema].truncate()
        datasource = self.datasources[TimeIntervalSchema]
        TimeIntervalSchema.test(self, datasource, docs)

    def test_interface_from_url(self):
        interface = get_interface('http://someserver.com/somepath')
        self.assertIsInstance(interface, RestInterface)

        interface = get_interface('https://someserver.com/somepath')
        self.assertIsInstance(interface, RestInterface)
