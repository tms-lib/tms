import math
from math import ceil, copysign, cos, exp, floor, log, pi, sin, sqrt
from statistics import median

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tms


# АКФ (Автокорреляция)
def autocorrelation(time_series, max_lag=None):
    """
    Рассчитывает автокорреляционную функцию (ACF) для временного ряда.

    Параметры:

    - time_series: np.ndarray, временной ряд
    - max_lag: int, максимальная задержка (лаг) для расчета ACF. Если None, то используется длина временного ряда.

    Возвращает:

    - acf_values: np.ndarray, значения автокорреляционной функции
    """

    if not isinstance(time_series, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    n = len(time_series)
    if max_lag is None:
        max_lag = n - 1

    acf_values = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        shifted_series = np.roll(time_series, shift=lag)
        acf_values[lag] = np.corrcoef(time_series[lag:], shifted_series[lag:])[0, 1]

    return tms.TimeSeries(acf_values)


# ЧАКФ (Частичная автокорреляция)
def partial_autocorrelation(time_series, max_lag=None):
    """
    Рассчитывает частичную автокорреляционную функцию (PACF) для временного ряда.

    Параметры:

    - time_series: np.ndarray, временной ряд
    - max_lag: int, максимальная задержка (лаг) для расчета PACF. Если None, то используется длина временного ряда.

    Возвращает:

    - pacf_values: np.ndarray, значения частичной автокорреляционной функции
    """

    if not isinstance(time_series, np.ndarray):
        raise ValueError("Подайте на вход временной ряд.")

    if max_lag is None:
        max_lag = len(time_series) - 1

    pacf_values = np.zeros(max_lag + 1)

    for k in range(1, max_lag + 1):
        y = time_series[k:]
        X = np.column_stack([time_series[:-k], np.ones(len(y))])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pacf_values[k] = beta[0]

    return tms.TimeSeries(pacf_values)


# Критерий Дарбина - Уотсона
def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b0 = y_mean - b1 * x_mean
    return b0, b1


def calculate_residuals(x, y, b0, b1):
    y_pred = b0 + b1 * x
    residuals = y - y_pred
    return residuals


def durbin_watson_statistic(time_series):
    """
    Рассчитывает статистику Дарбина-Уотсона для временного ряда.

    Параметры:

    - time_series: np.ndarray, временной ряд

    Возвращает:

    - d: float, Статистику Дарбина-Уотсона
    """
    n = len(time_series)
    x = np.arange(n)
    y = time_series

    b0, b1 = linear_regression(x, y)

    residuals = calculate_residuals(x, y, b0, b1)

    diff_resid = np.diff(residuals)

    numerator = float(np.sum(diff_resid**2))

    denominator = float(np.sum(residuals**2))

    d = numerator / denominator if denominator else None

    return d


def rectification(ts):
    new_ts = []
    m = np.percentile(ts, 35)

    for i, cur in enumerate(ts):
        if i == 0:
            new_ts.append(log(ts[i + 1], m))
        else:
            new_ts.append(log(cur, m))

    exp_new_ts = [abs(exp(el) - exp(1)) if el < 1 else exp(el) for el in new_ts]
    coefs = [
        abs(exp(exp_el / log_el) - exp(1)) for exp_el, log_el in zip(exp_new_ts, new_ts)
    ]
    return [el * k for el, k in zip(ts, coefs)]


def tan_to_degrees(tan):
    angle_radians = math.atan(tan)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def delta(i, j, r=50, p=2):
    if abs(i - j) > r:
        return 0
    return (1 - abs(i - j) / r) ** p


def simple_n(a, b):
    return (b - a) / (a + b)


def exp_n(a, b, sigma=1):
    return (1 - exp((-1) * (abs(a - b) ** 2) / (sigma * (abs(b) + 1) ** 2))) * copysign(
        1, a - b
    )


def kmean(_set, b, q=1):
    # Колмогоровское среднее
    # берем по модулю
    nom = sum([simple_n(abs(ai), abs(b)) ** q for ai in _set])
    return (nom / len(_set)) ** (1 / q)


def M_avg_weighted(ts, i, p):
    l = range(len(ts))
    _delta = [delta(i, j, p=p) for j in l]
    return sum([ts[j] * d for j, d in zip(l, _delta)]) / sum(_delta)


def w_energy(ts, i, p):
    l = range(len(ts))
    _delta = [delta(i, j, p=p) for j in l]
    return sum(
        abs(ts[j] - M_avg_weighted(ts, j, p=p)) * d for j, d in zip(l, _delta)
    ) / sum(_delta)


def energy_measure(ts, i, p):
    l = range(len(ts))
    ef = w_energy(ts, i, p)
    im_ef = [w_energy(ts, i, p) for i in l]
    mq = kmean(im_ef, i)
    return (ef - mq) / (ef + mq)


class ActivityMeasures:
    def __init__(self, ts, p=2, q=1, r=50):
        self.ts = ts
        self.l = list(range(len(ts)))
        self.p = p
        self.q = q
        self.r = r
        self.im_ef = []
        self._delta = []
        self._M_avg_weighted = []

    def delta(self, i, j):
        if abs(i - j) > self.r:
            return 0
        return (1 - abs(i - j) / self.r) ** self.p

    def M_avg_weighted(self):
        if not self._M_avg_weighted:
            self._M_avg_weighted = (
                sum([self.ts[j] * d for j, d in zip(self.l, self._delta[i])])
                / self._delta_sum[i]
            )

    def preprocess(self):
        if not self._delta:
            for i in self.l:
                self._delta.append([self.delta(i, j) for j in self.l])
            self._delta_sum = [sum(d) for d in self._delta]

        if not self._M_avg_weighted:
            for i in self.l:
                self._M_avg_weighted.append(
                    sum([self.ts[j] * d for j, d in zip(self.l, self._delta[i])])
                    / self._delta_sum[i]
                )

    def w_energy(self, i):
        res = (
            sum(
                abs(self.ts[j] - self._M_avg_weighted[j]) * d
                for j, d in zip(self.l, self._delta[i])
            )
            / self._delta_sum[i]
        )
        return res

    def energy_measure(self, i):
        if not self.im_ef:
            for i in self.l:
                self.im_ef.append(self.w_energy(i))
        ef = self.im_ef[i]
        mq = kmean(self.im_ef, i)
        return (ef - mq) / (ef + mq)

    def regres_diff_measure(self, i):
        if not self.im_ef:
            reg = RegressionProcessor(self.ts, self.p, self.r, self.l)
            reg.preprocess()
            betta = [reg.calc_b(i) for i in self.l]
            self.im_ef = [abs(xi - bi) for xi, bi in zip(self.ts, betta)]
        ef = self.im_ef[i]
        mq = kmean(self.im_ef, i)
        return (ef - mq) / (ef + mq)

    def regres_measure(self, i):
        if not self.im_ef:
            reg = RegressionProcessor(self.ts, self.p, self.r, self.l)
            reg.preprocess()
            self.im_ef = [abs(reg.calc_b(i)) for i in self.l]
        ef = self.im_ef[i]
        mq = kmean(self.im_ef, i)
        return (ef - mq) / (ef + mq)


def energy_calc(y, t, delta=200):
    """Функционал Энергия"""
    l, r = max(0, t - delta), min(len(y), t + delta)
    window = y[l:r]
    # В исходной формуле умножение идет на дельта, но на краях элементов меньше, чем t-delta:t+delta
    denom = len(window) + 1
    y_dashed = sum([yi for yi in window]) / denom
    energy = sum([(yi - y_dashed) ** 2 for yi in window])
    return energy


def len_func_calc(y, t, delta):
    """Функционал Длина"""
    l, r = max(0, t - delta), min(len(y), t + delta)
    window = list(y[l:r])
    shifted_window = window[1:] + [window[-1]]
    res = sum([abs(next - cur) for cur, next in zip(window, shifted_window)])
    return res


def oscillation_calc(y, t, delta):
    """Функционал Осцилляция"""
    l, r = max(0, t - delta), min(len(y), t + delta)
    window = list(y[l:r])
    res = max(window) - min(window)
    return res


def n_0(a, b):
    m = max(a, b)
    if m == 0:
        return 0
    return (b - a) / m


def psi(t, gamma):
    dif = t - gamma
    denom = 1 - gamma * copysign(1, dif)
    if denom == 0:
        return 0
    return dif / denom


def n(a, b, gamma=-0.5):
    return psi(n_0(a, b), gamma)


class AbstractProcessor:
    def __init__(self):
        pass

    def preprocess(self):
        pass

    def calc(self):
        pass


class MeasuresProcessor(AbstractProcessor):
    def __init__(self, x, delta, gamma, conjunction_type):
        self.x = x
        self.delta = delta
        self.gamma = gamma
        self.conjunction_type = conjunction_type

    def deviations_calc(self, ti):
        l, r = max(0, ti - self.delta), min(len(self.x), ti + self.delta)
        l_window = self.x[l:ti]
        r_window = self.x[ti:r]
        cur_val = self.x[ti]
        x_li = sum([(cur_val - xi) if xi > cur_val else 0 for xi in l_window])
        x_ls = sum([(xi - cur_val) if xi < cur_val else 0 for xi in l_window])
        x_ri = sum([(cur_val - xi) if xi > cur_val else 0 for xi in r_window])
        x_rs = sum([(xi - cur_val) if xi < cur_val else 0 for xi in r_window])
        return x_li, x_ls, x_ri, x_rs

    def regres_deviations_calc(
        self, ti, p=0.5, reg_window=20, include_first=True, add_rectification=False
    ):
        l, r = max(0, ti - self.delta), min(len(self.x), ti + self.delta)

        l_window = self.x[l : ti + 1]

        r_window = self.x[ti:r]

        cur_val = self.x[ti]

        if include_first:
            values = [xi for xi in l_window if round(xi, 2) > round(cur_val, 2)] + [
                cur_val
            ]
        else:
            values = [xi for xi in l_window if round(xi, 2) > round(cur_val, 2)]

        if add_rectification:
            values = rectification(values) if len(values) > 5 else values

        _len = len(values)
        t_values = range(_len)

        x_li = (
            calc_regres_alpha(
                values, p, reg_window, t_values, how="last", include_first=include_first
            )
            if len(values) > 5
            else None
        )

        if include_first:
            values = [xi for xi in l_window if round(xi, 2) < round(cur_val, 2)] + [
                cur_val
            ]
        else:
            values = [xi for xi in l_window if round(xi, 2) < round(cur_val, 2)]

        if add_rectification:
            values = rectification(values) if len(values) > 5 else values

        _len = len(values)
        t_values = range(_len)

        x_ls = (
            calc_regres_alpha(
                values, p, reg_window, t_values, how="last", include_first=include_first
            )
            if len(values) > 5
            else None
        )

        if include_first:
            values = [cur_val] + [
                xi for xi in r_window if round(xi, 2) > round(cur_val, 2)
            ]
        else:
            values = [xi for xi in r_window if round(xi, 2) > round(cur_val, 2)]

        if add_rectification:
            values = rectification(values) if len(values) > 5 else values

        _len = len(values)
        t_values = range(_len)

        x_ri = (
            calc_regres_alpha(
                values,
                p,
                reg_window,
                t_values,
                how="first",
                include_first=include_first,
            )
            if len(values) > 5
            else None
        )

        if include_first:
            values = [cur_val] + [
                xi for xi in r_window if round(xi, 2) < round(cur_val, 2)
            ]
        else:
            values = [xi for xi in r_window if round(xi, 2) < round(cur_val, 2)]

        if add_rectification:
            values = rectification(values) if len(values) > 5 else values

        _len = len(values)
        t_values = range(_len)
        x_rs = (
            calc_regres_alpha(
                values,
                p,
                reg_window,
                t_values,
                how="first",
                include_first=include_first,
            )
            if len(values) > 5
            else None
        )

        return x_li, x_ls, x_ri, x_rs

    def window_deviations_calc(self):
        self.window_deviations = [
            self.deviations_calc(ti)
            for ti in range(self.delta, len(self.x) - self.delta)
        ]

    def min_measures_calc(self, ti):
        x_li, x_ls, x_ri, x_rs = self.deviations_calc(ti)

        # предварительно считаем self.window_deviations

        sigma_li_plus = sum(
            [(wd[0] - x_li) if wd[0] > x_li else 0 for wd in self.window_deviations]
        )
        sigma_li_minus = sum(
            [(-wd[0] + x_li) if wd[0] < x_li else 0 for wd in self.window_deviations]
        )

        m_li = simple_n(sigma_li_minus, sigma_li_plus)

        sigma_ls_plus = sum(
            [(wd[1] - x_ls) if wd[1] > x_ls else 0 for wd in self.window_deviations]
        )
        sigma_ls_minus = sum(
            [(-wd[1] + x_ls) if wd[1] < x_ls else 0 for wd in self.window_deviations]
        )

        m_ls = simple_n(sigma_ls_minus, sigma_ls_plus)

        sigma_ri_plus = sum(
            [(wd[2] - x_ri) if wd[2] > x_ri else 0 for wd in self.window_deviations]
        )
        sigma_ri_minus = sum(
            [(-wd[2] + x_ri) if wd[2] < x_ri else 0 for wd in self.window_deviations]
        )

        m_ri = simple_n(sigma_ri_minus, sigma_ri_plus)

        sigma_rs_plus = sum(
            [(wd[3] - x_rs) if wd[3] > x_rs else 0 for wd in self.window_deviations]
        )
        sigma_rs_minus = sum(
            [(-wd[3] + x_rs) if wd[3] < x_rs else 0 for wd in self.window_deviations]
        )

        m_rs = simple_n(sigma_rs_minus, sigma_rs_plus)

        return m_li, m_ls, m_ri, m_rs

    def conjuction(self, a, b, c, d):
        if self.conjunction_type == "Умножение":
            return a * b * c * d
        elif self.conjunction_type == "Минимум":
            return min(a, b, c, d)

    def geom_measures_calc(self, ti):
        res = {}
        m_li, m_ls, m_ri, m_rs = self.min_measures_calc(ti)

        res["Фон"] = self.conjuction(-m_li, -m_ls, -m_ri, -m_rs)
        res["Старт горы"] = self.conjuction(-m_li, -m_ls, m_ri, -m_rs)
        res["Подъем"] = self.conjuction(-m_li, m_ls, m_ri, -m_rs)
        res["Пик"] = self.conjuction(-m_li, m_ls, -m_ri, m_rs)
        res["Спуск"] = self.conjuction(m_li, -m_ls, -m_ri, m_rs)
        res["Конец горы"] = self.conjuction(m_li, -m_ls, -m_ri, -m_rs)
        res["Впадина"] = self.conjuction(m_li, -m_ls, m_ri, -m_rs)
        res["Начало плоскогорья"] = self.conjuction(-m_li, m_ls, -m_ri, -m_rs)
        res["Конец плоскогорья"] = self.conjuction(-m_li, -m_ls, -m_ri, m_rs)
        res["Левая осцилляция"] = self.conjuction(m_li, m_ls, -m_ri, -m_rs)
        res["Правая осцилляция"] = self.conjuction(-m_li, -m_ls, m_ri, m_rs)
        res["Правая осцилляция с левым возрастанием"] = self.conjuction(
            -m_li, m_ls, m_ri, m_rs
        )
        res["Правая осцилляция с левым убыванием"] = self.conjuction(
            m_li, -m_ls, m_ri, m_rs
        )
        res["Левая осцилляция с правым убыванием"] = self.conjuction(
            m_li, m_ls, -m_ri, m_rs
        )
        res["Левая осцилляция с правым возрастанием"] = self.conjuction(
            m_li, m_ls, m_ri, -m_rs
        )
        res["Двусторонняя осцилляция"] = self.conjuction(m_li, m_ls, m_ri, m_rs)

        return res

    def preprocess(self):
        # считаем отклонения для всех точек в окне
        self.window_deviations_calc()

    def calc(self, ti):
        res = self.geom_measures_calc(ti)
        return res


def add_autocorr_noise(ts, mu=0.5):
    l = len(ts)
    res = [ts[0]]
    for el in ts[1:]:
        res.append(-res[-1] * 0.8 + mu * np.random.uniform(-1, 1, 1))
    return res


def add_small_noise(ts, d=0.5):
    return list(map(lambda x: x + 0.2 * np.random.normal(0, d, 1), ts))


color_map = {
    "Фон": "tab:blue",
    "Пик": "tab:red",
    "Старт горы": "tab:purple",
    "Конец горы": "tab:pink",
    "Спуск": "tab:green",
    "Подъем": "tab:olive",
    "Впадина": "tab:brown",
    "Начало плоскогорья": "tab:orange",
    "Конец плоскогорья": "tab:grey",
    "Левая осцилляция": "tab:cyan",
    "Правая осцилляция": "yellow",
    "Правая осцилляция с левым возрастанием": "darkseagreen",
    "Правая осцилляция с левым убыванием": "papayawhip",
    "Левая осцилляция с правым убыванием": "darkslategray",
    "Левая осцилляция с правым возрастанием": "navy",
    "Двусторонняя осцилляция": "black",
}

patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]


# функция построения графика по сегментам
def segment_plot(
    types_array,
    t,
    x,
    color_map,
    patches,
    title,
    labels=None,
    min_val=-1.1,
    max_val=1.1,
    step=0.1,
):
    cur_type = types_array[0]
    start = 0
    for ti, _type in zip(t, types_array):
        if _type == cur_type:
            continue
        end = min(ti + 1, max(t))
        # строим графики для каждого отдельного сегмента
        plt.plot(range(start, end), x[start:end], color=color_map[cur_type])
        start = ti
        cur_type = _type
    else:
        end = len(t)
        plt.plot(range(start, end), x[start:end], color=color_map[cur_type])

    plt.legend(handles=patches, loc=1)
    plt.title(title)
    plt.ylim((min_val, max_val))

    yticks = list(np.arange(min_val, max_val + step, step))
    ylabels = [round(y, 2) for y in yticks]
    plt.yticks(yticks, ylabels)

    if labels:
        xticks = [li for li in t if li % 10 == 0] + [200]
        xlabels = [round(ti, 2) for ti, li in zip(labels, t) if li % 10 == 0] + [1]
        plt.xticks(xticks, xlabels)


def check_type(x):
    if x >= 0.5:
        return "[0.5, 1]"
    elif x >= 0:
        return "[0, 0.5)"
    else:
        return "[-1, 0)"


def check_type2(x):
    if x <= -0.5:
        return "[-1, -0.5)"
    elif x <= 0:
        return "(-0.5, 0]"
    else:
        return "(0, 1]"


def create_df(measures, l):
    df = pd.DataFrame(measures)
    df.columns = ["mes"]
    df["type"] = df.mes.apply(check_type)
    df["l"] = l
    return df


def simple_plot(t, ts, title=""):
    plt.plot(t, ts)
    _max = max(ts)
    _min = min(ts)
    step = (_max - _min) / 20

    plt.ylim((_min - 1, _max + 1))

    xticks = [li for li in t if li * 100 % 5 < 1e-6] + [1]
    xlabels = [round(ti, 2) for ti, li in zip(t, t) if li * 100 % 5 < 1e-6] + [1]
    plt.xticks(xticks, xlabels)

    plt.title(title)


class RegressionProcessor(AbstractProcessor):
    def __init__(self, x, p, r, t):
        self.x = x
        self.p = p
        self.r = r
        self.t = t
        self.l = range(len(t))

    def delta(self, i, j):
        if abs(i - j) > self.r:
            return 0
        return (1 - abs(i - j) / self.r) ** self.p

    def calcDeltaMatrix(self):
        self.DeltaMatrix = {}
        for i in self.l:
            self.DeltaMatrix[i] = []
            for j in self.l:
                val = self.delta(i, j)
                self.DeltaMatrix[i].append(val)

    def a(self, i, j):
        delta_row = self.DeltaMatrix[i]
        nom = delta_row[j]
        denom = sum([delta_row[k] for k in self.l])
        return nom / denom

    def calcA(self):
        self.A = []
        for _, row in self.DeltaMatrix.items():
            denom = sum(row)
            row = list(map(lambda x: x / denom, row))
            self.A.append(row)

    def m(self, i, t=None):
        if not t:
            t = self.t
        return sum([self.A[i][j] * t[j] for j in self.l])

    def calcM(self):
        self.M = [self.m(i) for i in self.l]

    def d(self, i):
        t_squared = [ti**2 for ti in self.t]
        return self.m(i, t_squared) - self.M[i] ** 2

    def calcD(self):
        self.D = [self.d(i) for i in self.l]

    def preprocess(self):
        self.calcDeltaMatrix()
        self.calcA()
        self.calcM()
        self.calcD()

    def calc(self, i, include_first):
        if include_first:
            res = sum(
                [
                    self.A[0][j] * (self.x[j] - self.x[0]) * (self.t[j] - self.t[0])
                    for j in self.l
                ]
            ) / sum([self.A[0][j] * (self.t[j] - self.t[0]) ** 2 for j in self.l])
        else:
            return sum(
                [
                    self.A[i][j]
                    * (1 / self.D[i] * self.t[j] - 1 / self.D[i] * self.M[i])
                    * self.x[j]
                    for j in self.l
                ]
            )
        return res

    def calc_b(self, i):
        return sum(
            [
                self.A[i][j]
                * self.x[j]
                * (
                    1
                    - (self.M[i] - self.t[i])
                    * (1 / self.D[i] * self.t[j] - 1 / self.D[i] * self.M[i])
                )
                for j in self.l
            ]
        )


def calc_regres_alpha(x, p, r, t, how, include_first):
    if how == "last":
        i = 0
        x = x[::-1]
        mul = -1
    else:
        i = 0
        mul = 1
    reg = RegressionProcessor(x, p, r, t)
    reg.preprocess()
    return reg.calc(i, include_first=include_first) * mul


benches = {
    "plain": [1, 1, 1, 1, 1, 1, 1, 1],
    "rise_start": [1, 1, 15, None, 1, 1, 15, None],
    "descent_end": [15, None, 1, 1, 15, None, 1, 1],
    "rise_end": [None, 15, 1, 1, None, 15, 1, 1],
    "descent_start": [1, 1, None, 15, 1, 1, None, 15],
    "rise": [None, 15, 15, None, None, 15, 15, None],
    "descent": [15, None, None, 15, 15, None, None, 15],
    "peak": [None, 15, None, 15, None, 15, None, 15],
    "hollow": [15, None, 15, None, 15, None, 15, None],
    "left_fading_oscillation": [15, 15, 1, 1, 15, 15, 1, 1],
    "right_fading_oscillation": [1, 1, 15, 15, 1, 1, 15, 15],
    "right_fading_oscillation_with_left_rise": [None, 15, 15, 15, None, 15, 15, 15],
    "right_fading_oscillation_with_left_decrease": [15, None, 15, 15, 15, None, 15, 15],
    "left_fading_oscillation_with_right_rise": [15, 15, 15, None, 15, 15, 15, None],
    "left_fading_oscillation_with_right_decrease": [15, 15, None, 15, 15, 15, None, 15],
    "double_fading_oscillation": [15, 15, 15, 15, 15, 15, 15, 15],
    "left_uniform_oscillation": [15, 15, 1, 1, 1, 1, 1, 1],
    "right_uniform_oscillation": [1, 1, 15, 15, 1, 1, 1, 1],
    "double_uniform_oscillation": [15, 15, 15, 15, 1, 1, 1, 1],
}


def coalesce(*args):
    for el in args:
        if el:
            return el


def calc_distance(point, need_log=False, return_full=False):
    point = [abs(el) if el else None for el in point]
    res = {}
    if need_log:
        print(point)
    for bench, bench_angles in benches.items():

        tp = 0  # true positive - угол должен быть и он есть
        tn = 0  # true negative - угла не должно быть и его нет
        fp = 0  # false positive - угла не должно быть, а он есть (ложное срабатывание)
        fn = 0  # false negative - угол должен быть, а его нет (ложное отрицание)

        tp_max = 0
        fp_max = 0
        fn_max = 0

        tp_cnt = 0
        fp_cnt = 0
        fn_cnt = 0

        sum_tp = sum(
            [(b - p) ** 2 if b and p else 0 for b, p in zip(bench_angles, point)]
        )

        for b_angle, p_angle in zip(bench_angles, point):

            if b_angle is not None:
                tp_max += 15**2
                fn_max += 15**2
            else:
                fp_max += 15**2 if p_angle else 0

            if b_angle is not None and p_angle is not None:
                tp += (b_angle - p_angle) ** 2
                tp_cnt += 1

                if need_log:
                    print(f"tp: {tp}")

            elif b_angle is None and p_angle is None:
                if need_log:
                    print("tn")
                ...

            elif b_angle is None and p_angle is not None:
                fp += (0 - p_angle) ** 2
                fp_cnt += 1

                if need_log:
                    print(f"fp: {fp}")

            else:
                fn += 15**2
                fn_cnt += 1

                if need_log:
                    print(f"fn: {fn}")

        rel_tp, rel_fp, rel_fn = (
            tp / coalesce(tp_max, 1),
            fp / coalesce(fp_max, 1),
            fn / coalesce(fn_max, 1),
        )

        f1 = sqrt(rel_tp**2 + rel_fp**2 + rel_fn**2)

        if need_log:
            print(f"{bench}: {f1}")
            print(f"tp: {tp}; tn: {tn}; fp: {fp}; fn: {fn}")

        res[bench] = f1

    if not return_full:
        return min(res.items(), key=lambda x: x[1])
    else:
        return res.items()

# свертка
def convolve(ts, kernel):
    n = len(ts)
    k = len(kernel)
    
    result = [0] * n
    
    offset = k // 2
    
    for i in range(n):
        conv_sum = 0
        for j in range(k):
            ts_index = i + j - offset
            
            if 0 <= ts_index < n:
                conv_sum += ts[ts_index] * kernel[j]
        
        result[i] = conv_sum
    
    return result



import math


# Фурье
def dft(signal):
    N = len(signal)
    result = []

    for k in range(N):  # Для каждого выходного элемента
        real_part = 0
        imag_part = 0

        for n in range(N):  # Для каждого входного элемента
            angle = 2 * math.pi * k * n / N
            real_part += signal[n] * math.cos(angle)
            imag_part -= signal[n] * math.sin(angle)

        result.append(complex(real_part, imag_part))

    return result