import os
import tempfile
import unittest
import rframe

from typing import List
from loguru import logger
import pandas as pd


from hypothesis import assume, given, settings
from hypothesis import strategies as st
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
        datasource = self.table
        SimpleSchema.test(self, datasource, docs)

    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        self.table.truncate()
        datasource = self.table
        SimpleMultiIndexSchema.test(self, datasource, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None)
    def test_interpolated(self, docs: InterpolatingSchema):
        self.table.truncate()
        datasource = self.table
        InterpolatingSchema.test(self, datasource, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        self.table.truncate()
        datasource = self.table
        IntegerIntervalSchema.test(self, datasource, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_time_interval(self, docs: TimeIntervalSchema):
        self.table.truncate()
        datasource = self.table
        TimeIntervalSchema.test(self, datasource, docs)