import pytest

import tms


def test_acf_return_ts(ts):
    assert isinstance(tms.eda.autocorrelation(ts), tms.TimeSeries)


def test_pacf_return_ts(ts):
    assert isinstance(tms.eda.partial_autocorrelation(ts), tms.TimeSeries)


def test_durbin_return_digit(ts):
    res = tms.eda.durbin_watson_statistic(ts)
    assert isinstance(res, (float, int)) or res is None


def test_acf_all_lower_1(ts):
    assert max(tms.eda.autocorrelation(ts)) <= 1


def test_acf_all_greater_0(ts):
    assert min(tms.eda.autocorrelation(ts)) >= 0
