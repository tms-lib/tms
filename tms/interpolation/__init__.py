from .linear import linear_interpolate
from .neighbor import nearest_neighbor_interpolate
from .poly import polynomial_interpolate
from .spline import spline_interpolate


def interpolate(arr, _type, *args, **kwargs):
    """
    Основная функция для интерполяции временного ряда.

    Параметры:

    - arr: np.ndarray, временной ряд
    - _type: str, тип интерполяции: linear, poly, spline, neighbor

    """
    if _type == "linear":
        return linear_interpolate(arr, *args, **kwargs)
    elif _type == "poly":
        return polynomial_interpolate(arr, *args, **kwargs)
    elif _type == "spline":
        return spline_interpolate(arr, *args, **kwargs)
    elif _type == "neighbor":
        return nearest_neighbor_interpolate(arr, *args, **kwargs)
    else:
        raise ValueError(
            "Неизвестный тип интерполяции. Возможные значения: linear, poly, spline, neighbor."
        )
