import os
import unittest
from typing import List

import pandas as pd
import pymongo
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import rframe

from .test_schema import *


def mongo_uri_not_set():
    return "TEST_MONGO_URI" not in os.environ


@unittest.skipIf(mongo_uri_not_set(), "No access to test database")
class TestMongo(unittest.TestCase):
    """
    Test the Mongodb interface

    Requires write access to some pymongo server, the URI of which is to be set
    as an environment variable under:

        TEST_MONGO_URI

    """

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        uri = os.environ.get("TEST_MONGO_URI")
        db_name = "rframe"
        collection_name = "test"
        client = pymongo.MongoClient(uri)
        database = client[db_name]
        self.collection = database[collection_name]

    @given(st.builds(SimpleSchema))
    def test_insert(self, doc: SimpleSchema):
        self.collection.delete_many({})
        doc.save(self.collection)
        doc_found = doc.find_one(self.collection, **doc.index_labels)
        assert doc.same_values(doc_found)

    @given(st.lists(st.builds(SimpleSchema), unique_by=lambda x: x.index, min_size=1))
    def test_frame(self, docs: List[SimpleSchema]):
        self.collection.delete_many({})
        rf = rframe.RemoteFrame(SimpleSchema, self.collection)
        for doc in docs:
            doc.save(self.collection)
        df = pd.DataFrame([doc.dict() for doc in docs])
        df2 = rf.sel()
        assert len(df) == len(df2)

    @given(
        st.lists(
            st.builds(InterpolatingSchema).filter(lambda x: abs(x.index) < 2**7),
            unique_by=lambda x: x.index,
            min_size=2,
        )
    )
    def test_interpolated(self, docs: InterpolatingSchema):
        pass
