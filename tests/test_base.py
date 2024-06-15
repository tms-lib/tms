import numpy as np
import pytest

import tms


def pass_function(ts):
    return ts


def float_pow(x):
    return x**2.5


def test_timeseries_is_numpy(ts):
    assert isinstance(ts, np.ndarray)


def test_input_could_be_list():
    ts = tms.TimeSeries([1, 2, 3])
    assert isinstance(ts, tms.TimeSeries)


def test_input_could_be_tuple():
    ts = tms.TimeSeries((1, 2, 3))
    assert isinstance(ts, tms.TimeSeries)


def test_input_could_be_np_array():
    ts = tms.TimeSeries(np.array([1, 2, 3]))
    assert isinstance(ts, tms.TimeSeries)


def test_default_type_is_float():
    ts = tms.TimeSeries([1, 2, 3])
    assert ts.dtype == float


def test_override_type():
    ts = tms.TimeSeries([1, 2, 3], dtype=int)
    assert ts.dtype == int


def test_int_type_raises_error_with_none():
    with pytest.raises(Exception):
        ts = tms.TimeSeries([1, 2, 3, None, 5], dtype=int)


def test_add_returns_timeseries(ts):
    ts = ts.add(pass_function)
    assert isinstance(ts, tms.TimeSeries)


def test_add_returns_the_same_dtype():
    ts = tms.TimeSeries([1, 2, 3, 4, 5], dtype=int)
    _dtype1 = ts.dtype
    ts = ts.add(float_pow)
    _dtype2 = ts.dtype
    assert _dtype1 == _dtype2
