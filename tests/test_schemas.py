import datetime
from typing import ClassVar, Dict, List
import unittest
from pydantic import Field
from loguru import logger

import rframe
import pandas as pd
from rframe import BaseSchema, Index, InterpolatingIndex, Interval, IntervalIndex


from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rframe.schema import InsertionError, UpdateError
from rframe.utils import get_all_subclasses
from rframe.dispatchers import are_equal

# logger.disable("rframe")

int_indices = st.integers(min_value=1, max_value=1e8)
float_indices = st.floats(min_value=0, max_value=1e4, allow_nan=False)
float_values = st.floats(min_value=-1e8, max_value=1e8, allow_nan=False)


def round_datetime(dt):
    #
    return dt.replace(microsecond=int(dt.microsecond / 1000) * 1000, second=0)


datetimes = st.datetimes(
    min_value=datetime.datetime(2000, 1, 1, 0, 0),
    max_value=datetime.datetime(2232, 1, 1, 0, 0),
    allow_imaginary=False,
).map(round_datetime)

PERMISSIONS = {
    "insert": True,
    "update": False,
}


class BaseTestSchema(BaseSchema):
    def pre_insert(self, datasource):
        if not PERMISSIONS["insert"]:
            raise KeyError

    def pre_update(self, datasource, new):
        if not PERMISSIONS["update"]:
            raise KeyError

    @classmethod
    def field_strategies(cls) -> Dict[str, st.SearchStrategy]:
        return {}

    @classmethod
    def item_strategy(cls):
        return st.builds(cls, **cls.field_strategies())

    @classmethod
    def list_strategy(cls, **overrides):
        defaults = dict(
            unique_by=lambda x: x.index_labels_tuple,
            min_size=1,
            max_size=30,
        )
        kwargs = dict(defaults, **overrides)
        return st.lists(cls.item_strategy(), **kwargs)

    @classmethod
    def insert_data(
        cls, tester: unittest.TestCase, db, docs: List["InterpolatingSchema"]
    ):
        PERMISSIONS["insert"] = False
        with tester.assertRaises(InsertionError):
            db.insert(docs)

        PERMISSIONS["insert"] = True
        db.insert(docs)
 
        PERMISSIONS["update"] = False
        with tester.assertRaises(UpdateError):
            db.insert(docs)

    @classmethod
    def delete_data(
        cls, tester: unittest.TestCase, db, docs: List["BaseTestSchema"]
    ):
        db.delete(docs)
        tester.assertEqual(db.count(), 0)

    @classmethod
    def basic_tests(
        cls, tester: unittest.TestCase, db, docs: List["BaseTestSchema"]
    ):
        for doc in docs:
            doc_found = db.find_one(**doc.index_labels)
            assert doc.same_values(doc_found)

    @classmethod
    def frame_test(
        cls, tester: unittest.TestCase, db, docs: List["BaseTestSchema"]
    ):

        rf = rframe.RemoteFrame(cls, db.storage, lazy=True)

        # test RemoteFrame .at scalar lookup
        for doc in docs:
            for field, value in doc.column_values.items():
                assert are_equal(value, rf.at[doc.index_labels_tuple, field])

        # test RemoteSeries .at scalar lookup
        for doc in docs:
            for field, value in doc.column_values.items():
                assert are_equal(value, rf[field].at[doc.index_labels_tuple])

        df = pd.concat([doc.dframe() for doc in docs])

        tester.assertEqual(len(rf), len(df))

        df2 = rf.sel().df
        pd.testing.assert_frame_equal(df, df2, check_like=True)

        for field in rf.columns:
            max_val = df[field].max()
            assert are_equal(rf[field].max(), max_val)
            assert are_equal(rf.max(field), max_val)

            min_val = df[field].min()
            assert are_equal(rf[field].min(), min_val)
            assert are_equal(rf.min(field), min_val)

        n = max(1, min(len(df) // 2, 10))
        tester.assertEqual(n, len(rf.head(n)))

        for field in rf.columns:
            unique_vals = sorted(df[field].unique())
            assert are_equal(sorted(rf[field].unique()), unique_vals)
            assert are_equal(sorted(rf.unique(field)), unique_vals)

    @classmethod
    def test(cls, tester: unittest.TestCase, db, docs: List["BaseTestSchema"]):
        cls.insert_data(tester, db, docs)
        cls.basic_tests(tester, db, docs)
        cls.frame_test(tester, db, docs)
        cls.delete_data(tester, db, docs)


class SimpleSchema(BaseTestSchema):
    index_field: int = Index()
    value: float = Field()

    @classmethod
    def field_strategies(cls) -> Dict[str, st.SearchStrategy]:
        fields = dict(
            index_field=int_indices,
            value=float_values,
        )
        return fields


class SimpleMultiIndexSchema(BaseTestSchema):
    index1: int = Index()
    index2: str = Index(min_length=1)

    value1: float
    value2: str

    @classmethod
    def field_strategies(cls) -> Dict[str, st.SearchStrategy]:
        fields = dict(
            index1=int_indices,
            value1=float_values,
            value2=float_values,
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
        defaults = {"min_size": 3, "max_size": 10}
        kwargs = dict(defaults, **overrides)
        return super().list_strategy(**kwargs).map(sorted)

    @classmethod
    def field_strategies(cls) -> Dict[str, st.SearchStrategy]:
        fields = dict(
            index_field=float_indices,
            value=float_values,
        )
        return fields

    @classmethod
    def basic_tests(cls, tester, db, docs: List["InterpolatingSchema"]):

        for doc1, doc2 in zip(docs[:-1], docs[1:]):
            assume(1e-2 < abs(doc1.index_field - doc2.index_field) < 1e4)
            index = (doc1.index_field + doc2.index_field) / 2
            value = (doc1.value + doc2.value) / 2
            if value < 1e-2:
                continue
            doc = db.find_one(index_field=index)
            ratio = doc.value / value
            tester.assertAlmostEqual(ratio, 1, delta=1e-2)

    @classmethod
    def test(cls, tester, db, docs: List["BaseTestSchema"]):
        cls.insert_data(tester, db, docs)
        cls.basic_tests(tester, db, docs)
        cls.delete_data(tester, db, docs)


class IntervalTestSchema(BaseTestSchema):
    @classmethod
    def list_strategy(cls, **overrides):
        defaults = dict(
            unique_by=lambda x: x.index_field.left,
            min_size=2,
            max_size=10,
        )
        kwargs = dict(defaults, **overrides)
        return st.lists(cls.item_strategy(), **kwargs)

    @classmethod
    def basic_tests(cls, tester, db, docs: List["IntervalTestSchema"]):

        for doc in docs:
            # add half difference so it works for datatimes as well as ints
            half_diff = (doc.index_field.right - doc.index_field.left) / 2
            if half_diff < doc.index_field._resolution:
                continue
            index_val = doc.index_field.left + half_diff
            labels = doc.index_labels
            labels["index_field"] = index_val
            found_doc = db.find_one(**labels)
            assert found_doc.same_values(doc)
            assert found_doc.same_index(doc)

    @classmethod
    def test(cls, tester, db, docs: List["BaseTestSchema"]):
        cls.insert_data(tester, db, docs)
        cls.basic_tests(tester, db, docs)
        cls.frame_test(tester, db, docs)
        cls.delete_data(tester, db, docs)


@st.composite
def touching_intervals(draw, strategy, resolution):
    docs = draw(strategy)
    last = docs[-1].index_field.left + 10_000 * resolution
    last = min(last, docs[-1].index_field._max)

    borders = sorted([doc.index_field.left for doc in docs]) + [last]

    for doc, left, right in zip(docs, borders[:-1], borders[1:]):
        assume(right - left > resolution)

        # set the new boundaries
        doc.index_field = left, right

    return docs


class IntegerIntervalSchema(IntervalTestSchema):

    index_field: Interval[int] = IntervalIndex()
    value: float

    @classmethod
    def field_strategies(cls) -> Dict[str, st.SearchStrategy]:
        fields = dict(
            index_field=int_indices,
            value=float_values,
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
    def field_strategies(cls) -> Dict[str, st.SearchStrategy]:
        fields = dict(
            index_field=datetimes,
            value=float_values,
        )
        return fields

    @classmethod
    def list_strategy(cls, **overrides):
        resolution = datetime.timedelta(seconds=1)
        strategy = super().list_strategy(**overrides)
        strategy = touching_intervals(strategy, resolution=resolution)
        return strategy


TEST_SCHEMAS = {
    rframe.utils.camel_to_snake(klass.__name__): klass
    for klass in get_all_subclasses(BaseTestSchema)
}
