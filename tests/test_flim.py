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
            photons = pd.DataFrame(
                {"time_rel_pulse": np.arange(125), "Frames": 1, "Channel": 1}
            ).set_index(["Channel", "Frames"])
        summed_mem = di
        stack = di
        channels = pd.CategoricalIndex([1])
        out = PySightOutput(photons, summed_mem, stack, channels, shape, flim, dict())
        return LifetimeCalc(out, 1)

    return _generate_flim


@pytest.mark.skip
class TestFlim:
    shape = (10, 10, 10, 10)
    flim = True

    def test_get_photons(self, generate_flim):
        np.array_equal(
            generate_flim(self.shape, self.flim)._get_photons(), np.arange(125)
        )

    def test_aligned_hist(self, generate_flim):
        photons = np.arange(31)
        photons = np.concatenate((photons, [30, 30, 30, 29, 29, 28]))
        photons_df = pd.DataFrame(
            {"time_rel_pulse": photons, "Frames": 1, "Channel": 1}
        ).set_index(["Channel", "Frames"])
        hist = generate_flim(
            self.shape, self.flim, photons=photons_df
        )._gen_aligned_hist(photons)
        assert hist.argmax() == 0

    def test_lowest_bin(self, generate_flim):
        photons = np.linspace(0, 31, num=5, endpoint=False)
        photons = np.concatenate((photons, [30, 30, 30, 29, 29, 28]))
        photons = np.delete(photons, 0)
        photons_df = pd.DataFrame(
            {"time_rel_pulse": photons, "Frames": 1, "Channel": 1}
        ).set_index(["Channel", "Frames"])
        flim_obj = generate_flim(self.shape, self.flim, photons=photons_df)
        hist = flim_obj._gen_aligned_hist(photons)
        assert 2 == flim_obj._get_low_bin_idx(hist)
