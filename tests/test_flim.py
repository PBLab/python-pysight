import pytest
import pandas as pd
import numpy as np
import scipy.stats

from operator import mul
from functools import reduce

from pysight.post_processing.flim import *


@pytest.fixture
def mock_photons_df():
    channel = 1
    total_photon_count = 100_000
    total_frame_count = 100
    columns_per_frame = 100
    lines_per_frame = 10
    photons_per_frame = total_photon_count // total_frame_count
    frames = np.kron(
        np.arange(0, total_photon_count, columns_per_frame * lines_per_frame), np.ones(photons_per_frame, dtype=np.int64)
    )
    lines = np.kron(
        np.arange(0, total_photon_count, columns_per_frame),
        np.ones(columns_per_frame, dtype=np.int64),
    )
    abs_time = np.arange(total_photon_count)
    time_rel_line = abs_time - lines
    df = pd.DataFrame(
        {
            "Channel": channel,
            "Frames": frames,
            "Lines": lines,
            "abs_time": abs_time,
            "time_rel_line": time_rel_line,
        }
    )
    df = df.astype({"Channel": "category", "Frames": "category", "Lines": "category"})
    df = df.set_index(["Channel", "Frames", "Lines"])
    return df


def test_add_bins_to_df(mock_photons_df):
    line_edges = np.concatenate([mock_photons_df.index.get_level_values('Lines').unique().to_numpy(), [100_000]], axis=0)
    column_edges = np.arange(101)
    edges = [line_edges, column_edges]
    col_names = ["abs_time", "time_rel_line"]
    ret = add_bins_to_df(mock_photons_df.copy(), edges, col_names)
    mock_photons_df["bin_of_dim0"] = mock_photons_df.index.get_level_values("Lines").astype('int64') // 100
    mock_photons_df["bin_of_dim1"] = mock_photons_df["time_rel_line"]
    pd.testing.assert_frame_equal(ret, mock_photons_df)


def gen_exp_decay_points(shape=(256, 256), lambda_=35.0):
    """Generates 'shape' amount of randomly sampled points from an
    exp. decaying function with lambda=lambda_."""
    rv = scipy.stats.expon(scale=lambda_)
    num_pixels = reduce(mul, shape, 1)
    vals = rv.rvs(size=num_pixels).astype(np.uint16)
    return vals


def test_calculate_tau_per_image():
    decay_data = gen_exp_decay_points()
    tau = calc_lifetime(decay_data.ravel())
    assert tau == 35.0


@pytest.mark.skip
def test_per_frame_flim_calc():
    data = np.zeros((10, 256, 256), dtype=np.uint16)
    for frame_num in range(len(data)):
        data[frame_num] = gen_exp_decay_points(shape=data.shape[1:])
    lifetimes = calculate_lifetime_per_chunk(data, chunklen=1)
    assert np.allclose(lifetimes, np.array([35] * 10))


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
