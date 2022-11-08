import unittest

import pandas as pd

import rframe
from rframe.data_accessor import DataAccessor

from .test_schemas import SimpleSchema


class TestBasic(unittest.TestCase):
    """
    Test basic functionality

    """

    def setUp(self):
        index = list(range(100))
        values = list(range(100, 200))
        self.df = pd.DataFrame({"index_field": index, "value": values}).set_index(
            "index_field"
        )
        self.db = DataAccessor(SimpleSchema, self.df)

    def test_classmethods(self):
        SimpleSchema.get_index_fields()
        SimpleSchema.get_column_fields()

        labels, extra = SimpleSchema.extract_labels(index_field=1, other=7)
        self.assertDictEqual({"index_field": 1}, labels)
        self.assertDictEqual({"other": 7}, extra)

        index = SimpleSchema.index_for("index_field")
        self.assertIsInstance(index, rframe.Index)

    def test_summary_queries(self):
        
        assert self.db.max(fields="index_field") == 99
        assert self.db.min(fields="index_field") == 0
        assert self.db.max(fields="value") == 199
        assert self.db.min(fields="value") == 100
        assert self.db.count() == 100

        values = list(sorted(self.db.unique(fields="value")))
        self.assertListEqual(values, list(range(100, 200)))

        index = self.db.unique(fields="index_field")
        self.assertListEqual(index, list(range(100)))

    def test_queries(self):
        df2 = self.db.find_df()
        df2 = df2.sort_index()
        pd.testing.assert_frame_equal(self.df, df2, check_dtype=False)

        df2 = self.db.find_df(skip=2, limit=10)
        df2 = df2.sort_index()
        pd.testing.assert_frame_equal(self.df.iloc[2:12], df2, check_dtype=False)

        doc = self.db.find_one(index_field=1)
        self.assertEqual(doc.index_field, 1)
