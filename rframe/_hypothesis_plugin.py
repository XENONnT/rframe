"""
Register some custom strategies for hypothesis.
We need to constrain some of the types of the data we generate for testing.
 - Integers: size and datetime resolution are constrained by mongodb type system.
 - Datetimes: min-max are constrained by pandas (int64 representation of nanoseconds from epoch).
 - Floats: are limited by JSON encoding, cant be nan or inf.
 - Strings: we avoid zero length  as they can be very tricky when used as index labels.
 - Intervals: boundaries are constrained by the definition of the interval type.

This file is not imported by default, so it does not make hypothesis a hard dependency.
It is registered as a hypothesis plugin and thus is imported by the hypothesis package.
This approach is inspired by the pydantic hypothesis plugin.

"""
from .types import (
    MAX_INTEGER,
    MIN_DATETIME,
    MAX_TIMEDELTA,
    MAX_DATETIME,
    MIN_INTEGER,
    MIN_INTEGER_DELTA,
    MIN_TIMEDELTA,
    TimeInterval,
    IntegerInterval,
)
import datetime
from hypothesis import assume, strategies as st


st.register_type_strategy(float, st.floats(allow_nan=False, allow_infinity=False))
st.register_type_strategy(str, st.text(min_size=1, max_size=1000))

st.register_type_strategy(
    int, st.integers(min_value=-MAX_INTEGER, max_value=MAX_INTEGER)
)

# Register datetime strategy
def round_datetime(dt):
    # mongo resolution is 1000 microseconds
    return dt.replace(microsecond=int(dt.microsecond / 1000) * 1000)


datetimes_strategy = st.datetimes(
    min_value=MIN_DATETIME, max_value=MAX_DATETIME, allow_imaginary=False
).map(round_datetime)

timedeltas_strategy = st.timedeltas(
    min_value=MIN_TIMEDELTA,
    max_value=MAX_TIMEDELTA,
)

st.register_type_strategy(datetime.datetime, datetimes_strategy)
st.register_type_strategy(datetime.timedelta, timedeltas_strategy)


@st.composite
def make_interval(draw, type_, left_strategy, length_strategy):
    left = draw(left_strategy)
    length = draw(length_strategy)

    right = left + length

    assume(right <= type_._max)
    assume(left < type_._max)

    assume(right > type_._min)
    assume(left >= type_._min)

    return type_(left=left, right=right)


int_left_strategy = st.integers(min_value=MIN_INTEGER, max_value=MAX_INTEGER)

int_length_strategy = st.integers(
    min_value=MIN_INTEGER + MIN_INTEGER_DELTA * 2, max_value=MAX_INTEGER // 100
)

# Register the custom strategies for the two types of intervals TimeInterval, IntegerInterval.

st.register_type_strategy(
    IntegerInterval,
    make_interval(IntegerInterval, int_left_strategy, int_length_strategy),
)

interval_timedeltas_strategy = st.timedeltas(
    min_value=MIN_TIMEDELTA * 2,
    max_value=MAX_TIMEDELTA,
)

st.register_type_strategy(
    TimeInterval, make_interval(TimeInterval, datetimes_strategy, timedeltas_strategy)
)
