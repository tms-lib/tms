import numpy as np


def exponential_smoothing(arr, alpha, *args, **kwargs):
    """
    Выполняет экспоненциальное сглаживание временного ряда.

    Параметры:

    - arr: np.ndarray, временной ряд
    - alpha: float, коэффициент сглаживания (0 < alpha < 1)

    Возвращает:

    - smoothed_arr: np.ndarray, сглаженный временной ряд
    """

    if not isinstance(arr, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    n = len(arr)
    smoothed_arr = np.copy(arr)

    for idx in range(1, n):
        smoothed_arr[idx] = alpha * arr[idx] + (1 - alpha) * smoothed_arr[idx - 1]

    return smoothed_arr
