
import datetime
from typing import ClassVar, Dict, List
import unittest
from pydantic import Field
from loguru import logger

import rframe
import pandas as pd
from rframe import (BaseSchema, Index, InterpolatingIndex,
                    Interval, IntervalIndex)


from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rframe.schema import InsertionError, UpdateError
from rframe.utils import get_all_subclasses

logger.disable("rframe")

int_indices = st.integers(min_value=0, max_value=1e8)
float_indices = st.floats(min_value=0, max_value=1e4, allow_nan=False)
float_values = st.floats(min_value=-1e8, max_value=1e8, allow_nan=False)

def round_datetime(dt):
    return dt.replace(microsecond=0, second=0)

datetimes = st.datetimes(min_value=datetime.datetime(2000, 1, 1, 0, 0),
                         max_value=datetime.datetime(2232, 1, 1, 0, 0),
                         allow_imaginary=False).map(round_datetime)

PERMISSIONS = {
    'insert': True,
    'update': False,
}

class BaseTestSchema(BaseSchema):


    def pre_insert(self, datasource):
        if not PERMISSIONS['insert']:
            raise KeyError

    def pre_update(self, datasource, new):
        if not PERMISSIONS['update']:
            raise KeyError

    @classmethod
    def field_strategies(cls) -> Dict[str,st.SearchStrategy]:
        return {}

    @classmethod
    def item_strategy(cls):
        return st.builds(cls, **cls.field_strategies())
    
    @classmethod
    def list_strategy(cls, **overrides):
        defaults = dict(
            unique_by=lambda x: tuple(x.index_labels.values()),
            min_size=1, max_size=1000,
        )
        kwargs = dict(defaults, **overrides)
        return st.lists(cls.item_strategy(), **kwargs)

    @classmethod
    def insert_data(cls, tester: unittest.TestCase, datasource, docs: List['InterpolatingSchema']):
        PERMISSIONS['insert'] = False
        for doc in docs:
            with tester.assertRaises(InsertionError):
                doc.save(datasource)

        PERMISSIONS['insert'] = True
        for doc in docs:
            doc.save(datasource)

        PERMISSIONS['update'] = False
        for doc in docs:
            with tester.assertRaises(UpdateError):
                doc.save(datasource)


    @classmethod
    def basic_tests(cls, tester: unittest.TestCase, datasource, docs: List['BaseTestSchema']):
        for doc in docs:
            doc_found = cls.find_one(datasource, **doc.index_labels)
            assert doc.same_values(doc_found)

    @classmethod
    def frame_test(cls, tester: unittest.TestCase, datasource, docs: List['BaseTestSchema']):

        rf = rframe.RemoteFrame(cls, datasource)

        # test RemoteFrame .at scalar lookup
        for doc in docs:
            for field, value in doc.column_values.items():
                tester.assertEqual(value, rf.at[doc.index_labels_tuple, field])

        # test RemoteSeries .at scalar lookup
        for doc in docs:
            for field, value in doc.column_values.items():
                tester.assertEqual(value, rf[field].at[doc.index_labels_tuple])
        
        df = pd.concat([doc.to_pandas() for doc in docs])

        tester.assertEqual(len(rf), len(df))

        df2 = rf.sel().df
        pd.testing.assert_frame_equal(df, df2, check_like=True)

        for field in rf.columns:
            max_val = df[field].max()
            tester.assertEqual(rf[field].max(), max_val)
            tester.assertEqual(rf.max(field), max_val)

            min_val = df[field].min()
            tester.assertEqual(rf[field].min(),min_val)
            tester.assertEqual(rf.min(field),min_val)

        n = max(1, min(len(df)//2, 10) )
        tester.assertEqual(n, len(rf.head(n)))

        for field in rf.columns:
            unique_vals = sorted(df[field].unique())
            tester.assertEqual(sorted(rf[field].unique()), unique_vals)
            tester.assertEqual(sorted(rf.unique(field)), unique_vals)

    @classmethod
    def test(cls, tester: unittest.TestCase, datasource, docs: List['BaseTestSchema']):
        cls.insert_data(tester, datasource, docs)
        cls.basic_tests(tester, datasource, docs)
        cls.frame_test(tester, datasource, docs)

class SimpleSchema(BaseTestSchema):
    index_field: int = Index()
    value: float = Field()

    @classmethod
    def field_strategies(cls) -> Dict[str,st.SearchStrategy]:
        fields = dict(
            index_field = int_indices,
            value = float_values,
        )
        return fields


class SimpleMultiIndexSchema(BaseTestSchema):
    index1: int = Index()
    index2: str = Index(min_length=1)
  
    value1: float
    value2: str

    @classmethod
    def field_strategies(cls) -> Dict[str,st.SearchStrategy]:
        fields = dict(
            index1 = int_indices,
            value1 = float_values,
            value2 = float_values,
        )
        return fields

class AdvancedMultiIndexSchema(BaseTestSchema):
    index1: int = Index()
    index2: Interval[int] = IntervalIndex()
    index3: Interval[datetime.datetime] = IntervalIndex()
    index4: float = InterpolatingIndex()

    value: float


class InterpolatingSchema(BaseTestSchema):
    index_field: float = InterpolatingIndex()
    value: float

    @classmethod
    def list_strategy(cls, **overrides):
        defaults = {'min_size': 3 }
        kwargs = dict(defaults, **overrides)
        return super().list_strategy(**kwargs).map(lambda docs: sorted(docs, key=lambda x: x.index_field) )

    @classmethod
    def field_strategies(cls) -> Dict[str,st.SearchStrategy]:
        fields = dict(
            index_field = float_indices,
            value = float_values,
        )
        return fields


    @classmethod
    def basic_tests(cls, tester, datasource, docs: List['InterpolatingSchema']):

        for doc1,doc2 in zip(docs[:-1],docs[1:]):
            assume(1e-2 < abs(doc1.index_field - doc2.index_field) < 1e4)
            index = (doc1.index_field + doc2.index_field) / 2
            value = (doc1.value + doc2.value) / 2
            if value<1e-2:
                continue
            doc = cls.find_one(datasource, index_field=index)
            ratio = doc.value/value
            tester.assertAlmostEqual(ratio, 1, delta=1e-2)

    @classmethod
    def test(cls, tester, datasource, docs: List['BaseTestSchema']):
        cls.insert_data(tester, datasource, docs)
        cls.basic_tests(tester, datasource, docs)

class IntervalTestSchema(BaseTestSchema):

    @classmethod
    def list_strategy(cls, **overrides):
        defaults = dict(
            unique_by=lambda x: x.index_field.left,
            min_size=2, max_size=100
        )
        kwargs = dict(defaults, **overrides)
        return st.lists(cls.item_strategy(), **kwargs)

    @classmethod
    def basic_tests(cls, tester, datasource, docs: List['IntervalTestSchema']):
        
        for doc in docs:
            # add half difference so it works for datatimes as well as ints
            half_diff = (doc.index_field.right - doc.index_field.left) / 2
            index_val = doc.index_field.left + half_diff
            labels = doc.index_labels
            labels['index_field'] = index_val
            found_doc = cls.find_one(datasource, **labels)
            assert found_doc.same_values(doc)
            assert found_doc.same_index(doc)

    @classmethod
    def test(cls, tester, datasource, docs: List['BaseTestSchema']):
        cls.insert_data(tester, datasource, docs)
        cls.basic_tests(tester, datasource, docs)
        cls.frame_test(tester, datasource, docs)


@st.composite
def touching_intervals(draw, strategy, resolution):
    docs = draw(strategy)
    last = docs[-1].index_field.left + 100_000 * resolution
    
    borders = sorted([doc.index_field.left for doc in docs]) + [last]

    for doc,left,right in zip(docs, borders[:-1], borders[1:]):
        assume(right - left > resolution)

        doc.index_field.left = left
        doc.index_field.right = right
    return docs


class IntegerIntervalSchema(IntervalTestSchema):

    index_field: Interval[int] = IntervalIndex()
    value: float

    @classmethod
    def field_strategies(cls) -> Dict[str,st.SearchStrategy]:
        fields = dict(
            index_field = int_indices,
            value = float_values,
        )
        return fields

    @classmethod
    def list_strategy(cls, **overrides):
        strategy = super().list_strategy(**overrides)
        strategy = touching_intervals(strategy, resolution=1)
        return strategy



class TimeIntervalSchema(IntervalTestSchema):
    index_field: Interval[datetime.datetime] = IntervalIndex()
    value: float    

    @classmethod
    def field_strategies(cls) -> Dict[str,st.SearchStrategy]:
        fields = dict(
            index_field = datetimes,
            value = float_values,
        )
        return fields

    @classmethod
    def list_strategy(cls, **overrides):
        resolution = datetime.timedelta(seconds=1)
        strategy = super().list_strategy(**overrides)
        strategy = touching_intervals(strategy, resolution=resolution)
        return strategy


TEST_SCHEMAS = {rframe.utils.camel_to_snake(klass.__name__): klass
             for klass in get_all_subclasses(BaseTestSchema)}



