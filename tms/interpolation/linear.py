import numpy as np


def linear_interpolate(arr, left_width, right_width, *args, **kwargs):
    """
    Линейная интерполяция временного ряда.

    Параметры:

    - arr: np.ndarray, временной ряд
    - left_width: int, ширина левого окна
    - right_width: int, ширина правого окна

    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    missing_indices = np.where(np.isnan(arr))[0]

    for idx in missing_indices:
        left_width_counter, right_width_counter = left_width, right_width

        left_indexes = []
        right_indexes = []

        left_idx = idx - 1
        while left_width_counter >= 0:
            if not np.isnan(arr[left_idx]) and not left_idx < 0:
                left_indexes.append(left_idx)
            left_idx -= 1
            left_width_counter -= 1

        right_idx = idx + 1
        while right_width_counter >= 0:
            if not right_idx >= len(arr) and not np.isnan(arr[right_idx]):
                right_indexes.append(right_idx)
            right_idx += 1
            right_width_counter -= 1

        x = left_indexes + right_indexes
        y = arr[x]

        arr[idx] = np.interp(idx, x, y)

    return arr
