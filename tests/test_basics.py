import os
import unittest
from typing import List

import pandas as pd
import pymongo
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import rframe

from .test_schema import SimpleSchema


class TestBasic(unittest.TestCase):
    """
    Test basic functionality

    """
    def setUp(self):
        index = list(range(100))
        values = list(range(100, 200))
        self.df = pd.DataFrame({'index': index,
                                'values': values}
                                ).set_index('index')

    def test_classmethods(self):
        SimpleSchema.get_index_fields()
        SimpleSchema.get_column_fields()

        labels, extra = SimpleSchema.extract_labels(index=1, other=7)
        self.assertDictEqual({'index': 1}, labels)
        self.assertDictEqual({'other': 7}, extra)

        index = SimpleSchema.index_for('index')
        self.assertIsInstance(index, rframe.Index)

    def test_summary_queries(self):
        assert SimpleSchema.max(self.df, 'index') == 99
        assert SimpleSchema.min(self.df, 'index') == 0
        assert SimpleSchema.max(self.df, 'values') == 199
        assert SimpleSchema.min(self.df, 'values') == 100
        assert SimpleSchema.count(self.df) == 100

        values = list(sorted(SimpleSchema.unique(self.df, 'values')))
        self.assertListEqual(values, list(range(100, 200)))
        
        index = SimpleSchema.unique(self.df, 'index')
        self.assertListEqual(index, list(range(100)))
