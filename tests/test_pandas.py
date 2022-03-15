import os
import unittest
from typing import List

import pandas as pd
import pymongo
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import rframe

from .test_schemas import *


class TestPandas(unittest.TestCase):
    """
    Test the pandas interface

    """

    @given(SimpleSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_schema(self, docs: List[SimpleSchema]):
        datasource = docs[0].to_pandas().head(0)
        SimpleSchema.test(self, datasource, docs)
       
    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        datasource = docs[0].to_pandas().head(0)
        SimpleMultiIndexSchema.test(self, datasource, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None)
    def test_interpolated(self, docs: InterpolatingSchema):
        datasource = docs[0].to_pandas().head(0)
        InterpolatingSchema.test(self, datasource, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        datasource = docs[0].to_pandas().head(0)
        IntegerIntervalSchema.test(self, datasource, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None)
    def test_time_interval(self, docs: TimeIntervalSchema):
        datasource = docs[0].to_pandas().head(0)
        TimeIntervalSchema.test(self, datasource, docs)
