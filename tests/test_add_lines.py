import numpy as np
import pandas as pd
import pytest

from pysight.nd_hist_generator.line_signal_validators.add_bidir_lines import (
    add_bidir_lines,
)


@pytest.fixture
def even_data():
    lines = pd.DataFrame({"abs_time": np.arange(0, 11, 2)})
    data = dict(Lines=lines)
    return data


@pytest.fixture
def odd_data():
    lines = pd.DataFrame({"abs_time": np.arange(1, 12, 2)})
    data = dict(Lines=lines)
    return data


def test_add_even_lines(even_data):
    new_data = add_bidir_lines(even_data)
    correct = np.arange(0, 11, dtype=np.uint64)
    assert np.array_equal(new_data["Lines"].loc[:, "abs_time"].to_numpy(), correct)


def test_add_uneven_lines(odd_data):
    new_data = add_bidir_lines(odd_data)
    correct = np.arange(1, 12, dtype=np.uint64)
    assert np.array_equal(new_data["Lines"].loc[:, "abs_time"].to_numpy(), correct)
