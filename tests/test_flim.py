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


def gen_exp_decay_points(tau=35.0, bins=125, amp=100):
    """Generates exponential decay data with the given tau at the given 'bins'
    shape, with the given amplitude.
    """
    vals = scipy.signal.exponential(bins, 0, tau, False) * amp
    vals_to_hist = np.repeat(np.arange(bins), vals.astype(np.int64))
    return vals_to_hist


def test_calculate_tau_per_image():
    real_tau = 35.0
    decay_data = gen_exp_decay_points(tau=real_tau)
    returned_tau = calc_lifetime(decay_data)
    print(real_tau, returned_tau)
    assert np.isclose(real_tau, returned_tau, atol=1)


def test_add_frame_idx(mock_photons_df):
    frames = mock_photons_df.index.get_level_values("Frames").unique()
    # divide into frames with remainder
    data = add_downsample_frame_idx_to_df(mock_photons_df, 1, frames, 7)
    assert np.array_equal(np.unique(data.loc[:, 'frame_idx']), np.arange(0, 15))
    assert data[data['frame_idx'] == 0].count()['frame_idx'] == 7000
    assert data[data['frame_idx'] == 14].count()['frame_idx'] == 3000
    # divide into frames without remainder
    data = add_downsample_frame_idx_to_df(mock_photons_df, 1, frames, 10)
    assert np.array_equal(np.unique(data.loc[:, 'frame_idx']), np.arange(0, 10))
    assert data[data['frame_idx'] == 0].count()['frame_idx'] == 10000
    assert data[data['frame_idx'] == 9].count()['frame_idx'] == 10000


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
