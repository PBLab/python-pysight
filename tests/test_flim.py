import pytest
import pandas as pd
import numpy as np

from pysight.post_processing.flim import *


@pytest.fixture
def generate_flim():
    def _generate_flim(shape, flim, di=None, photons=None):
        if di is None:
            di = {1: np.array([1, 2, 3])}
        if photons is None:
            photons = pd.DataFrame({'time_rel_pulse': np.array([1, 2, 3])})
        summed_mem = di
        stack = di
        channels = pd.CategoricalIndex([1])
        out = PySightOutput(photons, summed_mem, stack, channels, shape, flim, dict())
        return LifetimeCalc(out, 1)
    return _generate_flim


class TestFlim:
    shape = (10, 10, 10)
    flim = True

    def test_get_photons(self, generate_flim):
        np.array_equal(generate_flim(self.shape, self.flim)._get_photons(), np.array([1, 2, 3]))
