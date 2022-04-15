
import datetime
from numbers import Number
import numpy as np
import toolz
from scipy.interpolate import interp1d

from .utils import are_equal, hashable_doc, singledispatch, unhashable_doc


def nn_interpolate(x, xs, ys):
    idx = np.argmin(np.abs(x - np.array(xs)))
    return ys[idx]


@singledispatch
def interpolate(x, xs, ys, kind="linear"):
    raise TypeError(f"Interpolation on type {type(x)} is not supported.")


@interpolate.register(Number)
def interpolate_number(x, xs, ys, kind="linear"):
    if all([isinstance(y, Number) for y in ys]):
        func = interp1d(
            xs, ys, fill_value=(ys[0], ys[-1]), bounds_error=False, kind=kind
        )
        return func(x).item()
    return nn_interpolate(x, xs, ys)


@interpolate.register(datetime.datetime)
def interpolate_datetime(x, xs, ys, kind="linear"):
    xs = [x.timestamp() for x in xs]
    x = x.timestamp()
    if all([isinstance(y, Number) for y in ys]):
        return interpolate_number(x, xs, ys, kind=kind)
    return nn_interpolate(x, xs, ys)

def interpolate_group(name, labels, group, fields, extrapolate=False):
    interpolated = []
    
    group = sorted(group, key=lambda x: x[name])
    xs = [d[name] for d in group]

    for label in labels:
        interp_doc = dict(unhashable_doc(group[-1]))
        
        if len(group) > 1 and label <= max(xs):
            for field in fields:
                ys = [d[field] for d in group]
                interp_doc[field] = interpolate(label, xs, ys)
            interp_doc[name] = label
            interpolated.append(interp_doc)
        elif extrapolate or are_equal(interp_doc[name], label):
            interp_doc[name] = label
            interpolated.append(interp_doc)
        
    return interpolated

def interpolate_records(name, labels, records,
                        groupby=(), extrapolate=False):
    
    if not records:
        return []

    records = [hashable_doc(record) for record in records]

    fields = [k for k in records[0] if k not in list(groupby)+[name]]
    
    if groupby:
        interpolated = []
        for _, group in toolz.groupby(groupby, records).items():
            if not group:
                continue
            group_docs = interpolate_group(name, labels, group, fields=fields, extrapolate=extrapolate)
    
            interpolated.extend(group_docs)
    else:
        return interpolate_group(name, labels, records, fields=fields)
        
    return interpolated

