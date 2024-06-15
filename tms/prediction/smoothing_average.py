import numpy as np


def moving_average_forecast(time_series, n):
    """
    Выполняет прогноз временного ряда с использованием метода скользящего среднего.

    Параметры:

    * time_series: Исходный временной ряд (numpy массив).
    * n: Размер окна скользящего среднего (целое число).

    Возвращает:

    * Прогнозированный временной ряд (numpy массив).
    """
    forecast = [time_series[0]]

    for i in range(len(time_series) - n + 1):
        window = time_series[i : i + n]
        forecast.append(np.mean(window))

    forecast.append(time_series[-1])

    last_value = forecast[-1]
    for i in range(n):
        forecast.append(last_value)

    return np.array(forecast)
