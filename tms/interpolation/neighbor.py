import numpy as np


def nearest_neighbor_interpolate(arr, left_width, right_width, *args, **kwargs):
    """
    Интерполяция методом ближайшего соседа.

    Параметры:

    - arr: np.ndarray, временной ряд
    - left_width: int, ширина левого окна
    - right_width: int, ширина правого окна

    """

    if not isinstance(arr, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    missing_indices = np.where(np.isnan(arr))[0]

    for idx in missing_indices:
        left_idx = idx - 1
        while left_width > 0 and np.isnan(arr[left_idx]):
            left_idx -= 1
            left_width -= 1

        right_idx = idx + 1
        while right_width > 0 and np.isnan(arr[right_idx]):
            right_idx += 1
            right_width -= 1

        # Ближайший сосед
        if abs(idx - left_idx) < abs(right_idx - idx):
            arr[idx] = arr[left_idx]
        else:
            arr[idx] = arr[right_idx]

    return arr
