import os
import sys

import pytest

from tms import *

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def ts():
    return TimeSeries([1, 2, 3, 4, 5, 6, 7, 8, 9])


@pytest.fixture
def missed_ts():
    data = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, np.nan, 8.0, 9.0])
    return data


@pytest.fixture
def non_smooth_ts():
    return TimeSeries([1, 2, 15, 4, 5, -5, 7, 8, 9])
