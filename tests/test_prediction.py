import pytest

import tms


def test_arima_prediction_return_tms(non_smooth_ts):
    isinstance(
        tms.prediction.predict(non_smooth_ts, "arima", [1, 2], 1, [1, 2], 3),
        tms.TimeSeries,
    )


def test_moving_average_prediction_return_tms(non_smooth_ts):
    isinstance(
        tms.prediction.predict(non_smooth_ts, "moving_average", 3), tms.TimeSeries
    )


def test_exponential_prediction_return_tms(non_smooth_ts):
    isinstance(
        tms.prediction.predict(non_smooth_ts, "exponential", 0.5, 3), tms.TimeSeries
    )


def test_holt_winters_prediction_return_tms(non_smooth_ts):
    isinstance(
        tms.prediction.predict(non_smooth_ts, "holt-winters", 5, 0.5, 0.5, 0.5, 3),
        tms.TimeSeries,
    )


def test_arima_prediction_length(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "arima", [1, 2], 1, [1, 2], 3)
    assert len(res) == (len(non_smooth_ts) + 3)


def test_moving_average_prediction_length(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "moving_average", 3)
    assert len(res) == (len(non_smooth_ts) + 3)


def test_exponential_prediction_length(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "exponential", 0.5, 3)
    assert len(res) == (len(non_smooth_ts) + 3)


def test_holt_winters_prediction_length(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "holt-winters", 5, 0.5, 0.5, 0.5, 3)
    assert len(res) == (len(non_smooth_ts) + 3)


def test_arima_prediction_contains_only_numbers(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "arima", [1, 2], 1, [1, 2], 3)
    prediction = res[-3:]
    assert sum([1 for x in prediction if isinstance(x, (int, float))]) == 3


def test_moving_average_prediction_contains_only_numbers(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "moving_average", 3)
    prediction = res[-3:]
    assert sum([1 for x in prediction if isinstance(x, (int, float))]) == 3


def test_exponential_prediction_contains_only_numbers(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "exponential", 0.5, 3)
    prediction = res[-3:]
    assert sum([1 for x in prediction if isinstance(x, (int, float))]) == 3


def test_holt_winters_prediction_contains_only_numbers(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "holt-winters", 5, 0.5, 0.5, 0.5, 3)
    prediction = res[-3:]
    assert sum([1 for x in prediction if isinstance(x, (int, float))]) == 3


def test_arima_prediction_adequate_result(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "arima", [1, 2], 1, [1, 2], 3)
    prediction = res[-3:]
    assert max(prediction) < 500


def test_moving_average_prediction_adequate_result(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "moving_average", 3)
    prediction = res[-3:]
    assert max(prediction) < 500


def test_exponential_prediction_adequate_result(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "exponential", 0.5, 3)
    prediction = res[-3:]
    assert max(prediction) < 500


def test_holt_winters_prediction_adequate_result(non_smooth_ts):
    res = tms.prediction.predict(non_smooth_ts, "holt-winters", 5, 0.5, 0.5, 0.5, 3)
    prediction = res[-3:]
    assert max(prediction) < 500
