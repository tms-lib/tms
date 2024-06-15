from collections import Counter

import numpy as np
from scipy.stats import chi2
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# Среднее значение
def mean(arr, weights=None):
    """Расчет среднего значения временного ряда arr"""
    return np.average(arr, weights=weights)


# Медианное значение
def median(arr):
    """Расчет медианы временного ряда arr"""
    return np.median(arr)


# Мода
def mode(arr):
    """Расчет моды временного ряда arr"""
    els = Counter(arr).most_common(0)
    if els:
        return els[0]
    else:
        return None


# Дисперсия
def variance(arr, weights=None):
    """Расчет дисперсии временного ряда arr"""
    return np.average((arr - mean(arr, weights)) ** 2, weights=weights)


# Стандартное отклонение
def std(arr, weights=None):
    """Расчет стандартного отклонения временного ряда arr"""
    return np.sqrt(variance(arr, weights))


# Колмогоровское среднее
def kolmogorov_mean(arr, power=2, weights=None):
    """
    Расчет колмогоровского среднего с учетом степени power и весовых коэффициентов.

    Параметры:

    * arr: list или numpy array, временной ряд
    * power: int: степень
    * weights: arr, массив коэффициентов
    """
    mean_value = mean(arr, weights)
    return np.average(np.abs(arr - mean_value) ** power, weights=weights) ** (1 / power)


# Максимум
def max(arr):
    """Расчет максимального значения временного ряда arr"""
    return np.max(arr)


# Минимум
def min(arr):
    """Расчет минимального значения временного ряда arr"""
    return np.min(arr)


# Коэффициент вариации
def variation(arr):
    """Расчет коэффициента вариации временного ряда arr"""
    return std(arr) / mean(arr)


# Перцентиль
def percentile(arr, q):
    """
    Расчет q-го перцентиля временного ряда.

    Параметры:

    * arr: list или numpy array, временной ряд
    * q: float: перцентиль
    """
    return np.percentile(arr, q)


# Коэффициент корреляции
def correlation(arr1, arr2):
    """
    Расчет корреляции между 2 временными рядами.

    Параметры:

    * arr1: list или numpy array, временной ряд
    * arr2: list или numpy array, временной ряд

    Возвращает:

    * correlation, float: значение корреляции
    """
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise ValueError("Подайте на вход 2 временных ряда.")

    if len(arr1) != len(arr2):
        raise ValueError("Подайте на вход 2 временных ряда одинаковой размерности.")

    covariance = np.cov(arr1, arr2)[0, 1]
    std_deviation1 = np.std(arr1)
    std_deviation2 = np.std(arr2)

    if std_deviation1 == 0 or std_deviation2 == 0:
        return 0

    correlation = covariance / (std_deviation1 * std_deviation2)
    return correlation


# Тест Дикки-Фуллера
def adf(arr):
    """
    Тест Дикки-Фуллера.

    Параметры:

    * arr: list или numpy array, временной ряд

    Возвращает:

    * p_value, float: значение p_value теста
    """
    diff_series = np.diff(arr)

    X = add_constant(arr[:-1])
    y = diff_series

    model = OLS(y, X)
    results = model.fit()

    adf_statistic = results.tvalues[1]

    p_value = 1 - results.pvalues[1]

    return p_value


# Тест Шапиро-Уилка
def shapiro_wilk(arr):
    """
    Тест Шапиро-Уилка.

    Параметры:

    * arr: list или numpy array, временной ряд

    Возвращает:

    * p_value, float: значение p_value теста
    """
    sorted_data = np.sort(arr)

    n = len(arr)
    m = int(np.floor(n / 2))
    a = -1.272 * (m ** (-0.315)) + 1.932 * (m ** (-0.5))

    if n % 2 == 0:
        w = np.sum(a * sorted_data[:m] + (1 - a) * sorted_data[-m:]) / np.sum(
            sorted_data**2
        )
    else:
        w = np.sum(
            list(a * sorted_data[: m + 1]) + list((1 - a) * sorted_data[-m:])
        ) / np.sum(sorted_data**2)

    statistic = (w**2) / (1 - (w**2 / n))

    p_value = 1.0 - statistic

    return p_value


# тест Колмогорова-Смирнова и соответствующие функции
def empirical_cdf(data):
    data_sorted = np.sort(data)
    n = len(data)
    return data_sorted, np.arange(1, n + 1) / n


def kolmogorov_smirnov_1sample(arr, cdf):
    """
    Одновыборочный тест Колмогорова-Смирнова.

    Параметры:

    * arr: list или numpy array, временной ряд
    * cdf: function, функция распределения (CDF) для сравнения.

    Возвращает:

    * p_value, float: значение p_value теста
    """
    data_sorted, ecdf = empirical_cdf(arr)
    cdf_values = cdf(data_sorted)
    d_stat = np.max(np.abs(ecdf - cdf_values))

    n = len(arr)
    p_value = np.exp(-2 * (d_stat**2) * n)

    return p_value


def kolmogorov_smirnov_2sample(arr1, arr2):
    """
    Двухвыборочный тест Колмогорова-Смирнова.

    Параметры:

    * arr1: list или numpy array, временной ряд
    * arr2: list или numpy array, временной ряд

    Возвращает:

    * p_value, float: значение p_value теста
    """
    data1_sorted, ecdf1 = empirical_cdf(arr1)
    data2_sorted, ecdf2 = empirical_cdf(arr2)

    all_data = np.sort(np.concatenate([data1_sorted, data2_sorted]))

    ecdf1_all = np.searchsorted(data1_sorted, all_data, side="right") / len(arr1)
    ecdf2_all = np.searchsorted(data2_sorted, all_data, side="right") / len(arr2)

    d_stat = np.max(np.abs(ecdf1_all - ecdf2_all))

    n1, n2 = len(arr1), len(arr2)
    n = n1 * n2 / (n1 + n2)
    p_value = np.exp(-2 * (d_stat**2) * n)

    return p_value


# тест Льюнга-Бокса
def autocorrelation(data, lag):
    n = len(data)
    mean = np.mean(data)
    numerator = np.sum((data[: n - lag] - mean) * (data[lag:] - mean))
    denominator = np.sum((data - mean) ** 2)
    return numerator / denominator


def ljung_box(arr, lags):
    """
    Выполняет тест Льюнга-Бокса для заданного временного ряда и количества лагов.

    Параметры:

    * arr: list или numpy array, временной ряд
    * lags: int: количество лагов

    Возвращает:

    * p_value, float: значение p_value теста
    """
    n = len(arr)
    q_stat = 0
    for lag in range(1, lags + 1):
        acf = autocorrelation(arr, lag)
        q_stat += (acf**2) / (n - lag)
    q_stat *= n * (n + 2)

    p_value = 1 - chi2.cdf(q_stat, lags)

    return p_value
