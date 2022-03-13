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

    def test_classmethods(self):
        SimpleSchema.get_index_fields()
        SimpleSchema.get_column_fields()

        labels, extra = SimpleSchema.extract_labels(index=1, other=7)
        self.assertDictEqual({'index': 1}, labels)
        self.assertDictEqual({'other': 7}, extra)

        index = SimpleSchema.index_for('index')
        self.assertIsInstance(index, rframe.Index)