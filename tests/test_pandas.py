import os
import tempfile
import unittest
from typing import List

import pandas as pd
import pymongo
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import rframe
from rframe.data_accessor import DataAccessor
from rframe.interfaces import get_interface, PandasInterface

from .test_schemas import *


class TestPandas(unittest.TestCase):
    """
    Test the pandas interface

    """

    @given(SimpleSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_schema(self, docs: List[SimpleSchema]):
        db = DataAccessor(SimpleSchema, docs[0].dframe().head(0))
        SimpleSchema.test(self, db, docs)

    @given(SimpleMultiIndexSchema.list_strategy())
    @settings(deadline=None)
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        db = DataAccessor(SimpleMultiIndexSchema, docs[0].dframe().head(0))
        SimpleMultiIndexSchema.test(self, db, docs)

    @given(InterpolatingSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_interpolated(self, docs: InterpolatingSchema):
        db = DataAccessor(InterpolatingSchema, docs[0].dframe().head(0))
        InterpolatingSchema.test(self, db, docs)

    @given(IntegerIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_integer_interval(self, docs: IntegerIntervalSchema):
        db = DataAccessor(IntegerIntervalSchema, docs[0].dframe().head(0))
        IntegerIntervalSchema.test(self, db, docs)

    @given(TimeIntervalSchema.list_strategy())
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_time_interval(self, docs: TimeIntervalSchema):
        db = DataAccessor(TimeIntervalSchema, docs[0].dframe().head(0))
        TimeIntervalSchema.test(self, db, docs)

    def test_interface_from_url(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            df.to_csv(f, index=False)
            f.seek(0)
            url = f.name
            interface = get_interface(url)
            self.assertIsInstance(interface, PandasInterface)

        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            df.to_pickle(f)
            f.seek(0)
            url = f.name
            interface = get_interface(url)
            self.assertIsInstance(interface, PandasInterface)
