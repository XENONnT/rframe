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

    @given(st.builds(SimpleSchema))
    def test_insert(self, doc: SimpleSchema):
        data = doc.pandas_dict()
        data['index'] = doc.index + 1
        df = pd.DataFrame([data]).set_index('index')

        doc.save(df)
        doc_found = doc.find_one(df, **doc.index_labels)
        assert doc.same_values(doc_found)

    @given(st.lists(st.builds(SimpleSchema),
                    unique_by=lambda x: x.index,
                    min_size=1, max_size=100))
    def test_frame(self, docs: List[SimpleSchema]):
        df = pd.DataFrame([doc.dict() for doc in docs]).set_index('index')

        rf = rframe.RemoteFrame(SimpleSchema, df)

        df2 = rf.sel()
        pd.testing.assert_frame_equal(df.sort_index(), df2.sort_index())

        pd.testing.assert_frame_equal(df.sort_index(), df2.sort_index())

        max_value = rf['value'].max()
        self.assertEqual(max_value, df['value'].max())

        min_value = rf['value'].min()
        self.assertEqual(min_value, df['value'].min())

        n = max(1, min(len(df)//2, 10) )
        self.assertEqual(n, len(rf.head(n)))

        self.assertEqual(sorted(rf['value'].unique()), sorted(df['value'].unique()))

    @given(st.lists(st.builds(SimpleMultiIndexSchema),
                    unique_by=lambda x: (x.index1,x.index2),
                    min_size=1, max_size=100))
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        df = pd.DataFrame([doc.dict() for doc in docs]).set_index(['index1', 'index2'])
        for doc in docs:
            doc_found = SimpleMultiIndexSchema.find_one(df, **doc.index_labels)
            assert doc.same_values(doc_found)

    # @given(
    #     st.lists(
    #         st.builds(InterpolatingSchema).filter(lambda x: abs(x.index) < 2**7),
    #         unique_by=lambda x: x.index,
    #         min_size=2,
    #         max_size=100,
    #     )
    # )
    # def test_interpolated(self, docs: InterpolatingSchema):
    #     pass
    
    # @given(
    #     st.lists(
    #         st.builds(IntegerIntervalSchema).filter(lambda x: abs(x.index.left) < 2**7),
    #         unique_by=lambda x: x.index.left,
    #         min_size=2,
    #         max_size=100,
    #     )
    # )
    # def test_integer_interval(self, docs: IntegerIntervalSchema):
    #     pass
    
    # @given(
    #     st.lists(
    #         st.builds(TimeIntervalSchema).filter(lambda x: abs(x.index.left) < 2**7),
    #         unique_by=lambda x: x.index.left,
    #         min_size=2,
    #         max_size=100,
    #     )
    # )
    # def test_time_interval(self, docs: TimeIntervalSchema):
    #     pass