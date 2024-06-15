import numpy as np
import statsmodels.api as sm
from scipy import optimize


def nonlinear_time_series_model(y):
    """
    Реализует нелинейную модель временного ряда с использованием логистической функции.

    Параметры:

    - y: np.ndarray, временной ряд

    Возвращает:

    - fitted_values: np.ndarray, предсказанные значения модели
    """

    time_index = np.arange(len(y))

    X = sm.add_constant(time_index)

    def logistic_function(params, x):
        return params[0] / (1 + np.exp(-params[1] * (x - params[2])))

    initial_params = [np.max(y), 1, np.mean(time_index)]

    loss_function = lambda params: np.sum(
        (y - logistic_function(params, time_index)) ** 2
    )

    result = optimize.minimize(loss_function, initial_params, method="Nelder-Mead")

    estimated_params = result.x

    fitted_values = logistic_function(estimated_params, time_index)

    return fitted_values
