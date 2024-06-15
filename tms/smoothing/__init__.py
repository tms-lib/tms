import tms

from .exponential import exponential_smoothing
from .moving_average import moving_average_smoothing
from .nonlinear import nonlinear_time_series_model


def smooth(arr, _type, *args, **kwargs):
    """
    Основная функция сглаживания временного ряда.

    Параметры:

    - arr: np.ndarray, временной ряд
    - _type: str, метод сглаживания: moving_average, exponential, nonlinear

    """
    if _type == "moving_average":
        return tms.TimeSeries(moving_average_smoothing(arr, *args, **kwargs))
    elif _type == "exponential":
        return tms.TimeSeries(exponential_smoothing(arr, *args, **kwargs))
    elif _type == "nonlinear":
        return tms.TimeSeries(nonlinear_time_series_model(arr, *args, **kwargs))
    else:
        raise ValueError(
            "Неизвестный тип сглаживания. Возможные значения: moving_average, exponential, nonlinear."
        )
