import os
import unittest
from typing import List

import pandas as pd
import pymongo
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import rframe
from rframe.schema import UpdateError

from .test_schemas import *

DB_NAME = "rframe_tests"
COLLECTION_NAME = "mongo_test"

MONGO_URI = os.environ.get("TEST_MONGO_URI")

@unittest.skipIf(MONGO_URI is None, "No access to test database")
class TestMongo(unittest.TestCase):
    """
    Test the Mongodb interface

    Requires write access to some pymongo server, the URI of which is to be set
    as an environment variable under:

        TEST_MONGO_URI

    """

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        
        client = pymongo.MongoClient(MONGO_URI)
        database = client[DB_NAME]
        self.collection = database[COLLECTION_NAME]

    def tearDown(self):
        client = pymongo.MongoClient(MONGO_URI)
        client.drop_database(DB_NAME)
    
    @given(st.builds(SimpleSchema))
    def test_insert(self, doc: SimpleSchema):
        self.collection.delete_many({})
        doc.save(self.collection)
        doc_found = doc.find_one(self.collection, **doc.index_labels)
        assert doc.same_values(doc_found)

    @given(st.lists(st.builds(SimpleSchema),
                    unique_by=lambda x: x.index,
                    min_size=1, max_size=100))
    def test_frame(self, docs: List[SimpleSchema]):
        self.collection.delete_many({})
        
        rf = rframe.RemoteFrame(SimpleSchema, self.collection)

        for doc in docs:
            doc.save(self.collection)

        df = pd.DataFrame([doc.dict() for doc in docs]).set_index('index')
        df2 = rf.sel()
        assert len(df) == len(df2)

        pd.testing.assert_frame_equal(df.sort_index(), df2.sort_index())

        self.assertEqual(rf['value'].max(), df['value'].max())

        self.assertEqual(rf['value'].min(), df['value'].min())

        n = max(1, min(len(df)//2, 10) )
        self.assertEqual(n, len(rf.head(n)))

        self.assertEqual(sorted(rf['value'].unique()), sorted(df['value'].unique()))

    @given(st.lists(st.builds(SimpleMultiIndexSchema),
                    unique_by=lambda x: (x.index1,x.index2),
                    min_size=1, max_size=100))
    def test_simple_multi_index(self, docs: List[SimpleMultiIndexSchema]):
        self.collection.delete_many({})
        for doc in docs:
            doc.save(self.collection)
        for doc in docs:
            doc_found = SimpleMultiIndexSchema.find_one(self.collection, **doc.index_labels)
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
