import rframe



import os
import unittest
from typing import List

import rframe
import pymongo
import pandas as pd

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from fastapi import FastAPI
from fastapi.testclient import TestClient


from .test_schema import *

def mongo_uri_not_set():
    return "TEST_MONGO_URI" not in os.environ


@unittest.skipIf(mongo_uri_not_set(), "No access to test database")
class TestRest(unittest.TestCase):
    """
    Test the Rest interface

    """
    

    def setUp(self):
        uri = os.environ.get("TEST_MONGO_URI")
        db_name = "rframe_tests"
        mongo_client = pymongo.MongoClient(uri)
        db = mongo_client[db_name]

        app = FastAPI()
        self.datasources = {}
        self.collections = {}
        for name, schema in SCHEMAS.items():
            collection = db[name]
            self.collections[schema] = collection
            router = rframe.SchemaRouter(
                    schema,
                    collection,
                    prefix=name,
                    )
            app.include_router(router)

        client = TestClient(app)
        for name, schema in SCHEMAS.items():
            source = rframe.RestClient('/'+name, client=client)
            self.datasources[schema] = source

    @given(st.builds(SimpleSchema))
    @settings(deadline=None)
    def test_insert(self, doc: SimpleSchema):
        self.collections[SimpleSchema].delete_many({})
        datasource = self.datasources[SimpleSchema]

        doc.save(datasource)
        doc_found = SimpleSchema.find_one(datasource, **doc.index_labels)
        self.assertIsNotNone(doc_found)
        assert doc.same_values(doc_found)

    @given(st.lists(st.builds(SimpleSchema), unique_by=lambda x: x.index, min_size=1, max_size=100))
    @settings(deadline=None)
    def test_frame(self, docs: List[SimpleSchema]):
        self.collections[SimpleSchema].delete_many({})
        datasource = self.datasources[SimpleSchema]

        rf = rframe.RemoteFrame(SimpleSchema, datasource)

        for doc in docs:
            doc.save(datasource)
        
        df = pd.DataFrame([doc.dict() for doc in docs]).set_index('index')
        
        df2 = rf.sel()
        assert isinstance(df2, pd.DataFrame)
        assert len(df) == len(df2)

        pd.testing.assert_frame_equal(df.sort_index(), df2.sort_index())

        max_value = rf['value'].max()
        self.assertEqual(max_value, df['value'].max())

        min_value = rf['value'].min()
        self.assertEqual(min_value, df['value'].min())

        n = max(1, min(len(df)//2, 10) )
        self.assertEqual(n, len(rf.head(n)))

        self.assertEqual(sorted(rf['value'].unique()), sorted(df['value'].unique()))

    # @given(
    #     st.lists(
    #         st.builds(InterpolatingSchema).filter(lambda x: abs(x.index) < 2**7),
    #         unique_by=lambda x: x.index,
    #         min_size=2,
    #     )
    # )
    # def test_interpolated(self, docs: InterpolatingSchema):
    #     pass
