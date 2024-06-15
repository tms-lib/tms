import numpy as np
import pytest

import tms


def test_linear_interpolate_return_timeseries(ts):
    assert isinstance(tms.interpolation.interpolate(ts, "linear", 1, 1), tms.TimeSeries)


def test_poly_interpolate_return_timeseries(ts):
    assert isinstance(
        tms.interpolation.interpolate(ts, "poly", 3, 1, 1), tms.TimeSeries
    )


def test_spline_interpolate_return_timeseries(ts):
    assert isinstance(tms.interpolation.interpolate(ts, "spline", 3, 3), tms.TimeSeries)


def test_neighbor_interpolate_return_timeseries(ts):
    assert isinstance(
        tms.interpolation.interpolate(ts, "neighbor", 1, 1), tms.TimeSeries
    )


def test_unknown_interpolation_type(ts):
    try:
        tms.interpolation.interpolate(ts, "unknown", 1, 1)
    except ValueError:
        assert True
    else:
        assert False


def test_poly_works(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "poly", 3, 2, 2)
    assert True


def test_linear_interpolation_not_return_nans(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "linear", 1, 1)
    assert len(res) == len(res[~np.isnan(res)])


def test_poly_interpolation_not_return_nans(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "poly", 3, 1, 1)
    assert len(res) == len(res[~np.isnan(res)])


def test_spline_interpolation_not_return_nans(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "spline", 3, 3)
    assert len(res) == len(res[~np.isnan(res)])


def test_neighbour_interpolation_not_return_nans(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "neighbor", 1, 1)
    assert len(res) == len(res[~np.isnan(res)])


def test_linear_values(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "linear", 1, 1)
    np.testing.assert_almost_equal(res, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_poly_values(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "poly", 3, 2, 3)
    np.testing.assert_almost_equal(res, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_neighbour_values(missed_ts):
    misses = len(missed_ts[np.isnan(missed_ts)])
    res = tms.interpolation.interpolate(missed_ts, "neighbor", 3, 3)
    assert sum(res - np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])) <= misses


def test_spline_values(missed_ts):
    res = tms.interpolation.interpolate(missed_ts, "spline", 3, 4)
    np.testing.assert_almost_equal(res, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
