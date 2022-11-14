import os
import unittest
from typing import List
from loguru import logger
import pandas as pd
import pymongo
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import rframe
from rframe.schema import UpdateError
from rframe.interfaces import get_interface, MongoInterface
from rframe.data_accessor import DataAccessor
from .test_schemas import *

DB_NAME = "rframe_tests"
COLLECTION_NAME = "mongo_test"

MONGO_URI = os.environ.get("TEST_MONGO_URI")


@unittest.skipIf(MONGO_URI is None, "No access to test database")
class TestMongo(unittest.TestCase):
    """
    Test the Mongodb interface

    Requires write access to some pymongo server, the URI of which is to be set
    as an environment variable under:

        TEST_MONGO_URI

    """

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        client = pymongo.MongoClient(MONGO_URI)
        database = client[DB_NAME]
        self.collection = database[COLLECTION_NAME]
        
    def tearDown(self):
        client = pymongo.MongoClient(MONGO_URI)
        client.drop_database(DB_NAME)

    @given(SimpleSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_schema(self, docs: List[SimpleSchema]):
        self.collection.delete_many({})
        db = DataAccessor(SimpleSchema, self.collection)
        SimpleSchema.test(self, db, docs)

    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        self.collection.delete_many({})
        db = DataAccessor(SimpleMultiIndexSchema, self.collection)
        SimpleMultiIndexSchema.test(self, db, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_interpolated(self, docs: InterpolatingSchema):
        self.collection.delete_many({})
        db = DataAccessor(InterpolatingSchema, self.collection)
        InterpolatingSchema.test(self, db, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        self.collection.delete_many({})
        db = DataAccessor(IntegerIntervalSchema, self.collection)
        IntegerIntervalSchema.test(self, db, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_time_interval(self, docs: TimeIntervalSchema):
        self.collection.delete_many({})
        db = DataAccessor(TimeIntervalSchema, self.collection)
        TimeIntervalSchema.test(self, db, docs)

    def test_interface_from_url(self):
        interface = get_interface(
            "mongodb://localhost", database="test", collection="test"
        )
        self.assertIsInstance(interface, rframe.interfaces.MongoInterface)

    def test_ensure_index(self):
        db = DataAccessor(AdvancedMultiIndexSchema, self.collection)
        db.initdb()
        name = "_".join([f"{name}_1" for name in AdvancedMultiIndexSchema.get_index_fields()])
        self.assertIn(name, self.collection.index_information())
