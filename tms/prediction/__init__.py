from .arima import arima_forecast
from .exponential import (
    exponential_smoothing_forecast,
    triple_exponential_smoothing_forecast,
)
from .smoothing_average import moving_average_forecast


def predict(arr, _type, *args, **kwargs):
    """
    Основная функция прогнозирования временного ряда.

    Параметры:

    - arr: np.ndarray, временной ряд
    - _type: str, метод прогноизрования: arima, moving_average, exponential, holt-winters

    """
    if _type == "arima":
        return arima_forecast(arr, *args, **kwargs)
    elif _type == "moving_average":
        return moving_average_forecast(arr, *args, **kwargs)
    elif _type == "exponential":
        return exponential_smoothing_forecast(arr, *args, **kwargs)
    elif _type == "holt-winters":
        return triple_exponential_smoothing_forecast(arr, *args, **kwargs)
    else:
        raise ValueError(
            "Неизвестный тип предсказания. Возможные значения: arima, moving_average_forecast, exponential, holt-winters."
        )
