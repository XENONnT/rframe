import unittest

import pandas as pd

import rframe

from .test_schemas import SimpleSchema


class TestBasic(unittest.TestCase):
    """
    Test basic functionality

    """
    def setUp(self):
        index = list(range(100))
        values = list(range(100, 200))
        self.df = pd.DataFrame({'index': index,
                                'value': values}
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
        assert SimpleSchema.max(self.df, 'value') == 199
        assert SimpleSchema.min(self.df, 'value') == 100
        assert SimpleSchema.count(self.df) == 100

        values = list(sorted(SimpleSchema.unique(self.df, 'value')))
        self.assertListEqual(values, list(range(100, 200)))
        
        index = SimpleSchema.unique(self.df, 'index')
        self.assertListEqual(index, list(range(100)))

    def test_queries(self):
        docs = SimpleSchema.find(self.df)
        df2 = pd.DataFrame([doc.pandas_dict() for doc in docs]).set_index('index')
        df2 = df2.sort_index()
        pd.testing.assert_frame_equal(self.df, df2, check_dtype=False)

        query = SimpleSchema.compile_query(self.df)
        docs = query.execute()
        df2 = pd.DataFrame(docs).set_index('index')
        df2 = df2.sort_index()
        pd.testing.assert_frame_equal(self.df, df2, check_dtype=False)

        doc = SimpleSchema.find_one(self.df, index=1)
        self.assertEqual(doc.index, 1)
