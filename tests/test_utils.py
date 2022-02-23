import hypothesis.strategies as st


@st.composite
def non_overlapping_interval_lists(draw, elements=st.datetimes(), min_size=2):
    elem = draw(st.lists(elements, unique=True, min_size=min_size * 2).map(sorted))
    return list(zip(elem[:-1:2], elem[1::2]))


@st.composite
def non_overlapping_interval_ranges(draw, elements=st.datetimes(), min_size=2):
    elem = draw(st.lists(elements, unique=True, min_size=min_size).map(sorted))
    return list(zip(elem[:-1], elem[1:]))
