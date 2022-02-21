import unittest
import os
import pymongo


def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


@unittest.skipIf(mongo_uri_not_set(), "No access to test database")
class TestMongo(unittest.TestCase):
    """
    Test the Mongodb interface 

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI


    """
    _run_test = True

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        uri = os.environ.get('TEST_MONGO_URI')
        db_name = 'test'
        collection_name = 'test'
        client = pymongo.MongoClient(uri)
        database = client[db_name]
        collection = database[collection_name]