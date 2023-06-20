import os
import tempfile
import unittest
import rframe

from typing import List
from loguru import logger
import pandas as pd


from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from rframe.data_accessor import DataAccessor
from tinydb import TinyDB, Query, where

from rframe.schema import UpdateError

from .test_schemas import *

DB_NAME = "tinydb_tests.json"
TABLE_NAME = "tinydb_test"


class TestTinyDB(unittest.TestCase):
    """
    Test the TinyDB interface

    """

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), DB_NAME)
        db = TinyDB(self.path)
        self.table = db.table(TABLE_NAME)

    def tearDown(self):
        os.remove(self.path)

    @given(SimpleSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_schema(self, docs: List[SimpleSchema]):
        self.table.truncate()
        db = DataAccessor(SimpleSchema, self.table)
        SimpleSchema.test(self, db, docs)

    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        self.table.truncate()
        db = DataAccessor(SimpleMultiIndexSchema, self.table)
        SimpleMultiIndexSchema.test(self, db, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_interpolated(self, docs: InterpolatingSchema):
        self.table.truncate()
        db = DataAccessor(InterpolatingSchema, self.table)
        InterpolatingSchema.test(self, db, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        self.table.truncate()
        db = DataAccessor(IntegerIntervalSchema, self.table)
        IntegerIntervalSchema.test(self, db, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_time_interval(self, docs: TimeIntervalSchema):
        self.table.truncate()
        db = DataAccessor(TimeIntervalSchema, self.table)
        TimeIntervalSchema.test(self, db, docs)
