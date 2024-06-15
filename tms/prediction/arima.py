import numpy as np


def generate_arima_data(phi, d, theta, n):
    """
    Генерирует временной ряд с использованием модели ARIMA.

    Параметры:
    - phi: list, коэффициенты авторегрессии
    - d: int, порядок интеграции (разность временного ряда)
    - theta: list, коэффициенты скользящего среднего
    - n: int, длина временного ряда

    Возвращает:
    - time_series: np.ndarray, сгенерированный временной ряд
    """

    p = len(phi)
    q = len(theta)

    # Генерация базового временного ряда
    time_series = np.cumsum(np.random.normal(size=n))

    # Применение авторегрессии (AR)
    for i in range(d, n):
        autoregressive_term = np.sum(phi * time_series[i - p : i])
        time_series[i] += autoregressive_term

    # Применение скользящего среднего (MA)
    for i in range(d, n):
        moving_average_term = np.sum(
            theta * (time_series[i - q : i] - np.mean(time_series[i - q : i]))
        )
        time_series[i] += moving_average_term

    return time_series


def arima_forecast(time_series, phi, d, theta, forecast_steps):
    """
    Прогнозирует временной ряд с использованием модели ARIMA.

    Параметры:

    - time_series: np.ndarray, временной ряд
    - phi: list, коэффициенты авторегрессии
    - d: int, порядок интеграции (разность временного ряда)
    - theta: list, коэффициенты скользящего среднего
    - forecast_steps: int, количество шагов для прогноза вперед

    Возвращает:

    - forecast: np.ndarray, прогнозируемые значения временного ряда
    """

    p = len(phi)
    q = len(theta)

    # Копируем исходный временной ряд
    forecast = np.copy(time_series)

    # Прогнозируем будущие значения
    for i in range(len(time_series), len(time_series) + forecast_steps):
        autoregressive_term = np.sum(phi * forecast[i - p : i])
        moving_average_term = np.sum(
            theta * (forecast[i - q : i] - np.mean(forecast[i - q : i]))
        )
        forecast = np.append(forecast, autoregressive_term + moving_average_term)

    return forecast
