import pytest

import tms


def test_moving_average_smoothing_return_tms(non_smooth_ts):
    assert isinstance(
        tms.smoothing.smooth(non_smooth_ts, "moving_average", 3, 3), tms.TimeSeries
    )


def test_exponential_smoothing_return_tms(non_smooth_ts):
    assert isinstance(
        tms.smoothing.smooth(non_smooth_ts, "exponential", 0.5), tms.TimeSeries
    )


def test_nonlinear_smoothing_return_tms(non_smooth_ts):
    assert isinstance(tms.smoothing.smooth(non_smooth_ts, "nonlinear"), tms.TimeSeries)


def test_undefined_smoothing_error(non_smooth_ts):
    try:
        isinstance(tms.smoothing.smooth(non_smooth_ts, "undefined"), tms.TimeSeries)
    except ValueError:
        assert True
    else:
        assert False


def test_moving_average_smoothing_result(non_smooth_ts):
    res = tms.smoothing.smooth(non_smooth_ts, "moving_average", 3, 3)
    assert len(res) == len(non_smooth_ts)
    assert abs(sum(res - non_smooth_ts)) < 10


def test_exponential_smoothing_result(non_smooth_ts):
    res = tms.smoothing.smooth(non_smooth_ts, "exponential", 0.5)
    assert len(res) == len(non_smooth_ts)
    assert abs(sum(res - non_smooth_ts)) < 10


def test_nonlinear_smoothing_result(non_smooth_ts):
    res = tms.smoothing.smooth(non_smooth_ts, "nonlinear")
    assert len(res) == len(non_smooth_ts)
    assert abs(sum(res - non_smooth_ts)) < 10
