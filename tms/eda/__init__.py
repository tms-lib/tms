import numpy as np

import tms


# АКФ (Автокорреляция)
def autocorrelation(time_series, max_lag=None):
    """
    Рассчитывает автокорреляционную функцию (ACF) для временного ряда.

    Параметры:

    - time_series: np.ndarray, временной ряд
    - max_lag: int, максимальная задержка (лаг) для расчета ACF. Если None, то используется длина временного ряда.

    Возвращает:

    - acf_values: np.ndarray, значения автокорреляционной функции
    """

    if not isinstance(time_series, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    n = len(time_series)
    if max_lag is None:
        max_lag = n - 1

    acf_values = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        shifted_series = np.roll(time_series, shift=lag)
        acf_values[lag] = np.corrcoef(time_series[lag:], shifted_series[lag:])[0, 1]

    return tms.TimeSeries(acf_values)


# ЧАКФ (Частичная автокорреляция)
def partial_autocorrelation(time_series, max_lag=None):
    """
    Рассчитывает частичную автокорреляционную функцию (PACF) для временного ряда.

    Параметры:

    - time_series: np.ndarray, временной ряд
    - max_lag: int, максимальная задержка (лаг) для расчета PACF. Если None, то используется длина временного ряда.

    Возвращает:

    - pacf_values: np.ndarray, значения частичной автокорреляционной функции
    """

    if not isinstance(time_series, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    if max_lag is None:
        max_lag = len(time_series) - 1

    pacf_values = np.zeros(max_lag + 1)

    for k in range(1, max_lag + 1):
        y = time_series[k:]
        X = np.column_stack([time_series[:-k], np.ones(len(y))])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pacf_values[k] = beta[0]

    return tms.TimeSeries(pacf_values)


# Критерий Дарбина - Уотсона
def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b0 = y_mean - b1 * x_mean
    return b0, b1


def calculate_residuals(x, y, b0, b1):
    y_pred = b0 + b1 * x
    residuals = y - y_pred
    return residuals


def durbin_watson_statistic(time_series):
    """
    Рассчитывает статистику Дарбина-Уотсона для временного ряда.

    Параметры:

    - time_series: np.ndarray, временной ряд

    Возвращает:

    - d: float, Статистику Дарбина-Уотсона
    """
    n = len(time_series)
    x = np.arange(n)
    y = time_series

    b0, b1 = linear_regression(x, y)

    residuals = calculate_residuals(x, y, b0, b1)

    diff_resid = np.diff(residuals)

    numerator = float(np.sum(diff_resid**2))

    denominator = float(np.sum(residuals**2))

    d = numerator / denominator if denominator else None

    return d
