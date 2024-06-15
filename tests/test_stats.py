import pytest
import scipy.stats as stats

import tms


def test_mean_return_digit_or_tms(ts):
    assert isinstance(tms.stats.mean(ts), (int, float, tms.TimeSeries))


def test_median_return_digit_or_tms(ts):
    assert isinstance(tms.stats.median(ts), (int, float, tms.TimeSeries))


def test_mode_return_digit_or_tms_or_None(ts):
    value = tms.stats.mode(ts)
    assert isinstance(value, (int, float, tms.TimeSeries)) or value is None


def test_variance_return_digit_or_tms(ts):
    assert isinstance(tms.stats.variance(ts), (int, float, tms.TimeSeries))


def test_std_return_digit_or_tms(ts):
    assert isinstance(tms.stats.std(ts), (int, float, tms.TimeSeries))


def test_kolmogorov_mean_return_digit_or_tms(ts):
    assert isinstance(tms.stats.kolmogorov_mean(ts), (int, float, tms.TimeSeries))


def test_max_return_digit_or_tms(ts):
    assert isinstance(tms.stats.max(ts), (int, float, tms.TimeSeries))


def test_min_return_digit_or_tms(ts):
    assert isinstance(tms.stats.min(ts), (int, float, tms.TimeSeries))


def test_variation_return_digit_or_tms(ts):
    assert isinstance(tms.stats.variation(ts), (int, float, tms.TimeSeries))


def test_percentile_return_digit_or_tms(ts):
    assert isinstance(tms.stats.percentile(ts, 0.4), (int, float, tms.TimeSeries))


def test_correlation_return_digit_or_tms(ts):
    assert isinstance(tms.stats.correlation(ts, ts), (int, float, tms.TimeSeries))


def test_adf_return_digit_or_tms(ts):
    assert isinstance(tms.stats.adf(ts), (int, float, tms.TimeSeries))


def test_shapiro_wilk_return_digit_or_tms(ts):
    assert isinstance(tms.stats.shapiro_wilk(ts), (int, float, tms.TimeSeries))


def test_kolmogorov_smirnov_1sample_return_digit_or_tms(ts):
    assert isinstance(
        tms.stats.kolmogorov_smirnov_1sample(
            ts, lambda x: stats.norm.cdf(x, loc=0, scale=1)
        ),
        (int, float, tms.TimeSeries),
    )


def test_kolmogorov_smirnov_2sample_return_digit_or_tms(ts):
    assert isinstance(
        tms.stats.kolmogorov_smirnov_2sample(ts, ts), (int, float, tms.TimeSeries)
    )


def test_ljung_box_return_digit_or_tms(ts):
    assert isinstance(tms.stats.ljung_box(ts, 3), (int, float, tms.TimeSeries))
