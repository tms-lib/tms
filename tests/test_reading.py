import os

import numpy as np
import pandas as pd
import pytest

import tms

# Путь к тестовым данным
xlsx_path = "test_data.xlsx"
csv_path = "test_data.csv"
txt_path = "test_data.txt"


# Тесты для создания объекта tms.TimeSeries
@pytest.mark.parametrize(
    "input_data",
    [
        np.array([1, 2, 3]),
        pd.Series([1, 2, 3]),
        [1, 2, 3],
        (1, 2, 3),
        {1, 2, 3},
    ],
)
def test_create_timeseries(input_data):
    ts = tms.TimeSeries(input_data)
    assert isinstance(ts, tms.TimeSeries)
    assert len(ts) == 3


# Дополнительные тесты для проверки создания tms.TimeSeries из файлов
def test_create_timeseries_from_excel():
    df = pd.DataFrame({"values": [1, 2, 3]})
    df.to_excel(xlsx_path, index=False)
    ts = tms.TimeSeries(xlsx_path)
    assert isinstance(ts, tms.TimeSeries)
    assert len(ts) == 3
    os.remove(xlsx_path)


def test_create_timeseries_from_csv():
    df = pd.DataFrame({"values": [1, 2, 3]})
    df.to_csv(csv_path, index=False)
    ts = tms.TimeSeries(csv_path)
    assert isinstance(ts, tms.TimeSeries)
    assert len(ts) == 3
    os.remove(csv_path)


def test_create_timeseries_from_txt():
    df = pd.DataFrame({"values": [1, 2, 3]})
    df.to_csv(txt_path, index=False, sep="\t")
    ts = tms.TimeSeries(txt_path)
    assert isinstance(ts, tms.TimeSeries)
    assert len(ts) == 3
    os.remove(txt_path)
