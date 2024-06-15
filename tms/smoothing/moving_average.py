import numpy as np


def moving_average_smoothing(
    arr, left_width, right_width, weights=None, *args, **kwargs
):
    """
    Выполняет сглаживание временного ряда скользящим средним.

    Параметры:

    - arr: np.ndarray, временной ряд
    - left_width: int, количество точек слева от текущей
    - right_width: int, количество точек справа от текущей
    - weights: np.ndarray, весовые коэффициенты для точек внутри окна

    Возвращает:

    - smoothed_arr: np.ndarray, сглаженный временной ряд
    """

    if not isinstance(arr, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    n = len(arr)
    smoothed_arr = np.copy(arr)

    for idx in range(n):
        left_idx = max(0, idx - left_width)
        right_idx = min(n, idx + right_width + 1)

        window = arr[left_idx:right_idx]

        if weights is None:
            smoothed_arr[idx] = np.mean(window)
        else:
            smoothed_arr[idx] = np.sum(weights * window) / np.sum(weights)

    return smoothed_arr
