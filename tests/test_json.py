import os
import tempfile
import unittest
import rframe

from typing import List
from loguru import logger
import pandas as pd

import fsspec

from hypothesis import assume, given, settings
from hypothesis import strategies as st
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
        datasource = []

        SimpleSchema.test(self, datasource, docs)

    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        datasource = []

        SimpleMultiIndexSchema.test(self, datasource, docs)

    #FIXME: This test is failing, but it's not clear why.
    # @given(InterpolatingSchema.list_strategy())
    # @settings(deadline=None)
    # def test_interpolated(self, docs: InterpolatingSchema):
    #     datasource = []

    #     InterpolatingSchema.test(self, datasource, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        datasource = []

        IntegerIntervalSchema.test(self, datasource, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_time_interval(self, docs: TimeIntervalSchema):
        datasource = []

        TimeIntervalSchema.test(self, datasource, docs)