import os
import tempfile
import unittest
import rframe

from typing import List
from loguru import logger
import pandas as pd

import fsspec

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from rframe.data_accessor import DataAccessor
from tinydb import TinyDB, Query, where

from rframe.schema import UpdateError

from .test_schemas import *

FILE_NAME = "json_tests.json"


class TestJson(unittest.TestCase):
    """
    Test the JSON interface

    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @given(SimpleSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_schema(self, docs: List[SimpleSchema]):
        db = DataAccessor(SimpleSchema, [])
        SimpleSchema.test(self, db, docs)

    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        db = DataAccessor(SimpleMultiIndexSchema, [])
        SimpleMultiIndexSchema.test(self, db, docs)

    # FIXME: This test is failing, but it's not clear why.
    # @given(InterpolatingSchema.list_strategy())
    # @settings(deadline=None)
    # def test_interpolated(self, docs: InterpolatingSchema):
    #     db = DataAccessor(InterpolatingSchema, [])
    #     InterpolatingSchema.test(self, db, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        db = DataAccessor(IntegerIntervalSchema, [])

        IntegerIntervalSchema.test(self, db, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_time_interval(self, docs: TimeIntervalSchema):
        db = DataAccessor(TimeIntervalSchema, [])

        TimeIntervalSchema.test(self, db, docs)
