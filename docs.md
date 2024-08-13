# tms

## tms.eda

### tms.eda.autocorrelation

Рассчитывает автокорреляционную функцию (ACF) для временного ряда.

Параметры:

* `time_series`: `np.ndarray`, временной ряд
* `max_lag`: `int`, максимальная задержка (лаг) для расчета ACF. Если `None`, то используется длина временного ряда.

Возвращает:

* `acf_values`: `np.ndarray`, значения автокорреляционной функции

### tms.eda.durbin_watson_statistic

Рассчитывает статистику Дарбина-Уотсона для временного ряда.

Параметры:

* `time_series`: `np.ndarray`, временной ряд

Возвращает:

* `d`: `float`, Статистику Дарбина-Уотсона

### tms.eda.partial_autocorrelation

Рассчитывает частичную автокорреляционную функцию (PACF) для временного ряда.

Параметры:

* `time_series`: `np.ndarray`, временной ряд
* `max_lag`: `int`, максимальная задержка (лаг) для расчета PACF. Если `None`, то используется длина временного ряда.

Возвращает:

* `pacf_values`: `np.ndarray`, значения частичной автокорреляционной функции







## tms.interpolation

### tms.interpolation.interpolate

Основная функция для интерполяции временного ряда.

Параметры:

* `arr`: `np.ndarray`, временной ряд
* `_type`: `str`, тип интерполяции: `linear`, `poly`, `spline`, `neighbor`

### tms.interpolation.linear.linear_interpolate

Линейная интерполяция временного ряда.

Параметры:

* `arr`: `np.ndarray`, временной ряд
* `left_width`: `int`, ширина левого окна
* `right_width`: `int`, ширина правого окна

### tms.interpolation.neighbor.nearest_neighbor_interpolate

Интерполяция методом ближайшего соседа.

Параметры:

* `arr`: `np.ndarray`, временной ряд
* `left_width`: `int`, ширина левого окна
* `right_width`: `int`, ширина правого окна

### tms.interpolation.poly.polynomial_interpolate

Полиномиальная интерполяция временного ряда.

Параметры:

* `arr`: `np.ndarray`, временной ряд
* `degree`: `int`, степень многочлена
* `left_width`: `int`, ширина левого окна
* `right_width`: `int`, ширина правого окна

### tms.interpolation.spline.spline_interpolate

Сплайновая интерполяция временного ряда.

Параметры:

* `arr`: `np.ndarray`, временной ряд
* `left_width`: `int`, ширина левого окна
* `right_width`: `int`, ширина правого окна
* `spline_kind`: `str`, тип сплайна






## tms.prediction

### tms.prediction.predict

Основная функция прогнозирования временного ряда.

Параметры:

* `arr`: `np.ndarray`, временной ряд
* `_type`: `str`, метод прогноизрования: `arima`, `moving_average`, `exponential`, `holt-winters`

### tms.prediction.arima.generate_arima_data

Генерирует временной ряд с использованием модели ARIMA.

Параметры:
- `phi`: `list`, коэффициенты авторегрессии
- `d`: `int`, порядок интеграции (разность временного ряда)
- `theta`: `list`, коэффициенты скользящего среднего
- `n`: `int`, длина временного ряда

Возвращает:
- `time_series`: `np.ndarray`, сгенерированный временной ряд

### tms.prediction.arima.arima_forecast

Прогнозирует временной ряд с использованием модели ARIMA.

Параметры:

- `time_series`: `np.ndarray`, временной ряд
- `phi`: `list`, коэффициенты авторегрессии
- `d`: `int`, порядок интеграции (разность временного ряда)
- `theta`: list, коэффициенты скользящего среднего
- `forecast_steps`: int, количество шагов для прогноза вперед

Возвращает:

- `forecast`: `np.ndarray`, прогнозируемые значения временного ряда

### tms.prediction.exponential.exponential_smoothing_forecast

Выполняет прогноз временного ряда с использованием метода экспоненциального сглаживания.

Параметры:

* `time_series`: Исходный временной ряд (`numpy` массив).
* `alpha`: Параметр сглаживания (число от 0 до 1).
* `forecast_periods`: Количество периодов для прогноза (целое число).

Возвращает:

* Прогнозированный временной ряд (`numpy` массив).

### tms.prediction.exponential.triple_exponential_smoothing_forecast

Прогноз временного ряда с использованием модели Хольта-Винтерса (тройного экспоненциального сглаживания).

Параметры:

* `time_series`: Исходный временной ряд (`numpy` массив).
* `season_length`: Длина сезонного цикла (целое число).
* `alpha`: Параметр сглаживания уровня (число от 0 до 1).
* `beta`: Параметр сглаживания тренда (число от 0 до 1).
* `gamma`: Параметр сглаживания сезонности (число от 0 до 1).
* `forecast_periods`: Количество периодов для прогноза (целое число).

Возвращает:

* Прогнозированный временной ряд (`numpy` массив).


### tms.prediction.smoothing_average.moving_average_forecast

Выполняет прогноз временного ряда с использованием метода скользящего среднего.

Параметры:

* `time_series`: Исходный временной ряд (`numpy` массив).
* `n`: Размер окна скользящего среднего (целое число).

Возвращает:

* Прогнозированный временной ряд (`numpy` массив).












## tms.smoothing

### tms.smoothing.smooth

Основная функция сглаживания временного ряда.

Параметры:

- `arr`: `np.ndarray`, временной ряд
- `_type`: `str`, метод сглаживания: `moving_average`, `exponential`, `nonlinear`

### tms.smoothing.exponential

Выполняет экспоненциальное сглаживание временного ряда.

Параметры:

- `arr`: `np.ndarray`, временной ряд
- `alpha`: `float`, коэффициент сглаживания (`0 < alpha < 1`)

Возвращает:

- `smoothed_arr`: `np.ndarray`, сглаженный временной ряд

### tms.smoothing.moving_average

Выполняет сглаживание временного ряда скользящим средним.

Параметры:

- `arr`: `np.ndarray`, временной ряд
- `left_width`: `int`, количество точек слева от текущей
- `right_width`: `int`, количество точек справа от текущей
- `weights`: `np.ndarray`, весовые коэффициенты для точек внутри окна

Возвращает:

- `smoothed_arr`: `np.ndarray`, сглаженный временной ряд

### tms.smoothing.nonlinear

Реализует нелинейную модель временного ряда с использованием логистической функции.

Параметры:

- `y`: `np.ndarray`, временной ряд

Возвращает:

- `fitted_values`: `np.ndarray`, предсказанные значения модели






## tms.stats

### tms.stats.mean

Расчет среднего значения временного ряда `arr`

### tms.stats.median

Расчет медианы временного ряда `arr`

### tms.stats.mode

Расчет моды временного ряда `arr`

### tms.stats.variance

Расчет дисперсии временного ряда `arr` с учетом массива весов `weights`

### tms.stats.std

Расчет стандартного отклонения временного ряда `arr` с учетом массива весов `weights`

### tms.stats.kolmogorov_mean

Расчет колмогоровского среднего с учетом степени power и весовых коэффициентов.

Параметры:

* `arr`: `list` или `numpy array`, временной ряд
* `power`: `int`: степень
* `weights`: `arr`, массив коэффициентов

### tms.stats.max

Расчет максимального значения временного ряда `arr`

### tms.stats.min

Расчет минимального значения временного ряда `arr`

### tms.stats.variation

Расчет коэффициента вариации временного ряда `arr`

### tms.stats.percentile

Расчет q-го перцентиля временного ряда.

Параметры:

* `arr`: `list` или `numpy array`, временной ряд
* `q`: `float`: перцентиль

### tms.stats.correlation

Расчет корреляции между 2 временными рядами.

Параметры:

* `arr1`: `list` или `numpy array`, временной ряд
* `arr2`: `list` или `numpy array`, временной ряд

Возвращает:

* `correlation`, `float`: значение корреляции

### tms.stats.adf

Тест Дикки-Фуллера.

Параметры:

* `arr`: `list` или `numpy array`, временной ряд

Возвращает:

* `p_value`, `float`: значение `p_value` теста

### tms.stats.shapiro_wilk

Тест Шапиро-Уилка.

Параметры:

* `arr`: `list` или `numpy array`, временной ряд

Возвращает:

* `p_value`, `float`: значение `p_value` теста

### tms.stats.kolmogorov_smirnov_1sample

Одновыборочный тест Колмогорова-Смирнова.

Параметры:

* `arr`: `list` или `numpy array`, временной ряд
* `cdf`: `function`, функция распределения (CDF) для сравнения.

Возвращает:

* `p_value`, `float`: значение `p_value` теста

### tms.stats.kolmogorov_smirnov_2sample

Двухвыборочный тест Колмогорова-Смирнова.

Параметры:

* `arr1`: `list` или `numpy array`, временной ряд
* `arr2`: `list` или `numpy array`, временной ряд

Возвращает:

* `p_value`, `float`: значение `p_value` теста

### tms.stats.ljung_box

Выполняет тест Льюнга-Бокса для заданного временного ряда и количества лагов.

Параметры:

* `arr`: `list` или `numpy array`, временной ряд
* `lags`: `int`: количество лагов

Возвращает:

* `p_value`, `float`: значение `p_value` теста

