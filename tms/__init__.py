import warnings

import numpy as np
import pandas as pd

from . import eda, interpolation, prediction, smoothing, stats

warnings.filterwarnings("ignore")


class TimeSeries(np.ndarray):
    """Базовый класс TimeSeries, который является основным строительным блоком библиотеки.
    
    Класс является наследником основного класса библиотеки Numpy - `np.ndarray`, что обеспечивает совместимость со всеми бибилотеками анализа данных.
    """

    def __new__(cls, input_data, dtype=float):
        """С помощью метода `__new__` создается объект класса `TimeSeries`, который является наследником `np.ndarray`.

        Параметры:
        
        * input_data (array_like): Данные временного ряда - столбец датафрейма pandas, массив numpy, файл xlsx/csv, данные из txt, список, кортеж, множество питон.
        * dtype (data-type, optional): Ожидаемый тип данных. Defaults to float.
        """
        if isinstance(input_data, pd.Series):
            obj = np.asarray(input_data.values, dtype=dtype).view(cls)
        elif isinstance(input_data, np.ndarray):
            obj = np.asarray(input_data, dtype=dtype).view(cls)
        elif isinstance(input_data, (list, tuple)):
            obj = np.asarray(input_data, dtype=dtype).view(cls)
        elif isinstance(input_data, set):
            obj = np.asarray(list(input_data), dtype=dtype).view(cls)
        elif isinstance(input_data, str):
            if input_data.endswith(".xlsx"):
                df = pd.read_excel(input_data)
                obj = np.asarray(df.iloc[:, 0].values, dtype=dtype).view(cls)
            elif input_data.endswith(".csv"):
                df = pd.read_csv(input_data)
                obj = np.asarray(df.iloc[:, 0].values, dtype=dtype).view(cls)
            elif input_data.endswith(".txt"):
                df = pd.read_csv(input_data, delimiter="\t")
                obj = np.asarray(df.iloc[:, 0].values, dtype=dtype).view(cls)
            else:
                raise ValueError(
                    "Неподдерживаемый формат файла. Поддерживаемые форматы: xlsx, csv, txt."
                )
        else:
            raise ValueError("Неподдерживаемый формат данных")

        return obj

    def __init__(self, input_data, dtype=float):
        self.dtype = dtype

    def add(self, func, *args, **kwargs):
        """Кастомный метод `add` позволяет "добавлять" к текущему временному ряду новые функции.
        Этот метод позволяет реализовать "принцип слоев", который лежит в основе библиотеки.

        Параметры:
        
        * func (function): Функция, которую нужно применить к исходному временному ряду
        * `*args`: Произвольные позиционные аргументы, которые будут переданы в функцию.
        * `**kwargs`: Произвольные именованные аргументы, которые будут переданы в функцию.

        Возвращает:
        
        * TimeSeries: Возвращается объект класса `TimeSeries`, который является результатом применения указанной функции к исходному временному ряду.
        """
        return TimeSeries(func(self, *args, **kwargs), self.dtype)
