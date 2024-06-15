import numpy as np


def exponential_smoothing_forecast(time_series, alpha, forecast_periods):
    """
    Выполняет прогноз временного ряда с использованием метода экспоненциального сглаживания.

    Параметры:

    * time_series: Исходный временной ряд (numpy массив).
    * alpha: Параметр сглаживания (число от 0 до 1).
    * forecast_periods: Количество периодов для прогноза (целое число).

    Возвращает:

    * Прогнозированный временной ряд (numpy массив).
    """
    forecast = []
    forecast.append(time_series[0])

    for t in range(1, len(time_series)):
        forecast.append(alpha * time_series[t] + (1 - alpha) * forecast[t - 1])

    last_value = forecast[-1]
    for _ in range(forecast_periods):
        forecast.append(alpha * last_value + (1 - alpha) * forecast[-1])

    return np.array(forecast)


def initial_seasonal_components(time_series, season_length):
    initial_level = np.mean(time_series[:season_length])
    initial_trend = (
        np.mean(time_series[season_length : 2 * season_length])
        - np.mean(time_series[:season_length])
    ) / season_length
    initial_seasonal = [time_series[i] - initial_level for i in range(season_length)]
    return initial_level, initial_trend, initial_seasonal


def triple_exponential_smoothing_forecast(
    time_series, season_length, alpha, beta, gamma, forecast_periods
):
    """
    Прогноз временного ряда с использованием модели Хольта-Винтерса (тройного экспоненциального сглаживания).

    Параметры:

    * time_series: Исходный временной ряд (numpy массив).
    * season_length: Длина сезонного цикла (целое число).
    * alpha: Параметр сглаживания уровня (число от 0 до 1).
    * beta: Параметр сглаживания тренда (число от 0 до 1).
    * gamma: Параметр сглаживания сезонности (число от 0 до 1).
    * forecast_periods: Количество периодов для прогноза (целое число).

    Возвращает:

    * Прогнозированный временной ряд (numpy массив).

    """
    forecast = []
    level, trend, seasonal = initial_seasonal_components(time_series, season_length)
    last_level = level

    for i in range(len(time_series) + forecast_periods):
        if i == 0:
            forecast.append(time_series[0])
            continue

        if i >= len(time_series):
            m = i - len(time_series) + 1
            forecast.append((level + m * trend) + seasonal[i % season_length])
        else:
            value = time_series[i]
            last_level, level = level, alpha * (value - seasonal[i % season_length]) + (
                1 - alpha
            ) * (last_level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonal[i % season_length] = (
                gamma * (value - level) + (1 - gamma) * seasonal[i % season_length]
            )
            forecast.append(level + trend + seasonal[i % season_length])

    return np.array(forecast)
