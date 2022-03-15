import rframe



import os
import unittest
from typing import List

import rframe
import pymongo
import pandas as pd

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from fastapi import FastAPI
from fastapi.testclient import TestClient


from .test_schemas import *

def mongo_uri_not_set():
    return "TEST_MONGO_URI" not in os.environ


@unittest.skipIf(mongo_uri_not_set(), "No access to test database")
class TestRest(unittest.TestCase):
    """
    Test the Rest interface

    """

    def setUp(self):
        uri = os.environ.get("TEST_MONGO_URI")
        db_name = "rframe_tests"
        mongo_client = pymongo.MongoClient(uri)
        db = mongo_client[db_name]

        app = FastAPI()
        self.datasources = {}
        self.collections = {}
        for name, schema in TEST_SCHEMAS.items():
            collection = db[name]
            self.collections[schema] = collection
            router = rframe.SchemaRouter(
                    schema,
                    collection,
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
        self.collections[SimpleSchema].delete_many({})
        datasource = self.datasources[SimpleSchema]
        SimpleSchema.test(self, datasource, docs)
       
    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        self.collections[SimpleMultiIndexSchema].delete_many({})
        datasource = self.datasources[SimpleMultiIndexSchema]
        SimpleMultiIndexSchema.test(self, datasource, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None)
    def test_interpolated(self, docs: InterpolatingSchema):
        self.collections[InterpolatingSchema].delete_many({})
        datasource = self.datasources[InterpolatingSchema]
        InterpolatingSchema.test(self, datasource, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        self.collections[IntegerIntervalSchema].delete_many({})
        datasource = self.datasources[IntegerIntervalSchema]
        IntegerIntervalSchema.test(self, datasource, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_time_interval(self, docs: TimeIntervalSchema):
        self.collections[TimeIntervalSchema].delete_many({})
        datasource = self.datasources[TimeIntervalSchema]
        TimeIntervalSchema.test(self, datasource, docs)
