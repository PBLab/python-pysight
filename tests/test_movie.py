from unittest import TestCase
from pysight.nd_hist_generator.movie import *
from pysight.nd_hist_generator.volume_gen import *
import pandas as pd
import numpy as np


def gen_data_df(frame_num=10, line_num=1000, end=100_000):
    """
    Mock data for tests.
    Returns:
        df - The full DataFrame
        frames only
        lines only
        x pixels
        y pixels
    """
    photons = np.arange(0, end, dtype=np.uint64)
    channel = np.ones_like(photons)
    lines = np.linspace(0, end, num=line_num, endpoint=False, dtype=np.uint64)
    x_pix = int(len(photons) / len(lines))
    ones_lines = np.ones((1, int(len(photons) / len(lines))), dtype=np.uint64)
    frames = np.linspace(0, end, num=frame_num, dtype=np.uint64, endpoint=False)
    frames_ser = pd.Series(frames)
    ones_frames = np.ones((1, int(len(photons) / len(frames))), dtype=np.uint64)
    lines = (np.atleast_2d(lines).T @ ones_lines).ravel()
    frames = (np.atleast_2d(frames).T @ ones_frames).ravel()
    assert len(lines) == len(frames) == len(photons)

    df = pd.DataFrame(
        {"abs_time": photons, "time_rel_line": photons - lines, "Lines": lines, "Frames": frames, "Channel": channel,}
    )
    df["Channel"] = df["Channel"].astype("category")
    df.set_index(["Channel", "Frames", "Lines"], drop=True, inplace=True)
    y_pix = x_pix
    return df, frames_ser, pd.Series(np.unique(lines)), x_pix, y_pix


class TestMovies(TestCase):
    frame_num = 10
    end = 1000
    line_num = 100
    data, frames, lines, x_pix, y_pix = gen_data_df(frame_num=frame_num, line_num=line_num, end=end)
    data_shape = (frame_num, x_pix, y_pix)
    volgen = VolumeGenerator(frames, data_shape)
    fr = volgen.create_frame_slices()
    movie = Movie(
        data=data,
        lines=lines,
        data_shape=data_shape,
        outputs={"memory": True},
        line_delta=int(lines.diff().mean()),
        fill_frac=100.0,
        bidir=True,
        frame_slices=fr,
        frames=frames,
        frames_per_chunk=volgen.frames_per_chunk,
    )
    movie.run()

    def test_all_pipeline_basic(self):
        self.assertTrue(np.all(self.movie.stack[1].ravel() == np.ones((self.end,), dtype=np.uint8)))

    def test_baseline_outputs(self):
        during, end = self.movie._Movie__determine_outputs()
        self.assertTrue(len(during) == 1)
        self.assertTrue(len(end) == 1)
        during, end = str(during), str(end)
        self.assertTrue("create_memory_output" in during)
        self.assertTrue("convert_list_to_arr" in end)

    def test_slice_df(self):
        sl = slice(0, 23000)
        movie = Movie(
            self.data,
            self.lines,
            data_shape=self.data_shape,
            outputs={"memory": True},
            line_delta=int(self.lines.diff().mean()),
            fill_frac=100.0,
            bidir=True,
            frames=self.frames,
            frame_slices=(slice(1) for n in range(2)),
        )
        di = movie._Movie__slice_df(sl)
        self.assertTrue(1 in di[0].keys())
        self.assertSequenceEqual((di[0][1].shape, di[1]), ((1000, 2), 10))

    def test_single_slice_df(self):
        sl = slice(0, 0)
        movie = Movie(
            self.data,
            self.lines,
            data_shape=self.data_shape,
            outputs={"memory": True},
            line_delta=int(self.lines.diff().mean()),
            fill_frac=100.0,
            bidir=True,
            frames=self.frames,
            frame_slices=(slice(1) for n in range(2)),
        )
        di = movie._Movie__slice_df(sl)
        self.assertTrue(1 in di[0].keys())
        self.assertSequenceEqual((di[0][1].shape, di[1]), ((100, 2), 1))


class TestMyHist:
    """Tests for my own implementation of an histogram."""

    def test_basic_indices(self):
        data = [np.array([5, 15, 25])]
        bins = [np.array([0, 10, 20, 30])]
        hist = HistWithIndex(data, bins)
        idx, _ = hist._get_indices_for_photons()
        np.testing.assert_equal(idx, np.array([1, 2, 3]))

    def test_out_of_bounds(self):
        data = [np.array([-1, 5, 30, 40])]
        bins = [np.array([0, 10, 20, 30])]
        hist = HistWithIndex(data, bins)
        idx, _ = hist._get_indices_for_photons()
        np.testing.assert_equal(idx, np.array([0, 1, 3, 4]))

    def test_indices_on_edge(self):
        data = [np.array([0, 5, 20, 30, 40])]
        bins = [np.array([0, 10, 20, 30])]
        hist = HistWithIndex(data, bins)
        idx, _ = hist._get_indices_for_photons()
        np.testing.assert_equal(idx, np.array([1, 1, 3, 3, 4]))

    def test_multidim(self):
        data = [np.array([0, 5, 20, 30, 40]), np.array([100, 200, 200, 300, 400])]
        bins = [np.array([0, 10, 20, 30]), np.array([100, 150, 200])]
        hist = HistWithIndex(data, bins)
        idx, _ = hist._get_indices_for_photons()
        np.testing.assert_equal(idx, np.array([5, 6, 14, 15, 19]))

    def test_hist_populate_basic(self):
        data = [np.array([5, 15, 25])]
        bins = [np.array([0, 10, 20, 30])]
        hist = HistWithIndex(data, bins)
        hist.run()
        np.testing.assert_equal(hist.hist_photons, np.array([1, 1, 1]))

    def test_myhist_against_histdd(self):
        data = [np.array([5, 15, 25])]
        bins = [np.array([0, 10, 20, 30])]
        hist = HistWithIndex(data, bins)
        hist.run()
        np.testing.assert_equal(hist.hist_photons, np.histogramdd(data, bins)[0])

    def test_my_multidim_against_histdd(self):
        data = [np.array([0, 5, 20, 30, 40]), np.array([100, 200, 200, 300, 400]), np.array([13, 14, 15, 16, 16])]
        bins = [np.array([0, 10, 20, 30]), np.array([100, 150, 200]), np.array([10, 20])]
        hist = HistWithIndex(data, bins)
        hist.run()
        histdd = np.histogramdd(data, bins)[0]
        np.testing.assert_equal(hist.hist_photons, histdd)


class TestFlimCalc:
    """A test suite for the FLIM calculation."""

    def test_simple_runner(self):
        data = np.array([5, 6, 15, 25])
        indices = np.array([0, 0, 1, 2])
        fl = FlimCalc(data, indices)
        fl.run()
        np.testing.assert_equal(np.array([5.5, 15, 25]), fl.hist_arrivals.to_numpy()[:, 1].ravel())

    def test_decay_borders_basic(self):
        hist = np.array([10, 20, 10, 5, 4, 3, 2, 10])
        peaks = np.array([1])
        props = {"peak_heights": np.array([20])}
        decay, _max, _min = find_decay_borders(hist, peaks, props)
        np.testing.assert_equal(decay, np.array([20, 10, 5, 4, 3, 2]))
        np.testing.assert_equal(_max, np.array([20]))
        np.testing.assert_equal(_min, np.array([2]))

    def test_decay_borders_min_at_end(self):
        hist = np.array([10, 20, 10, 5, 4, 3, 2])
        peaks = np.array([1])
        props = {"peak_heights": np.array([20])}
        decay, _max, _min = find_decay_borders(hist, peaks, props)
        np.testing.assert_equal(decay, np.array([20, 10, 5, 4, 3, 2]))
        np.testing.assert_equal(_max, np.array([20]))
        np.testing.assert_equal(_min, np.array([2]))

    def test_decay_borders_peak_at_start(self):
        hist = np.array([15, 10, 5, 4, 3, 2])
        peaks = np.array([0])
        props = {"peak_heights": np.array([15])}
        decay, _max, _min = find_decay_borders(hist, peaks, props)
        np.testing.assert_equal(decay, np.array([15, 10, 5, 4, 3, 2]))
        np.testing.assert_equal(_max, np.array([15]))
        np.testing.assert_equal(_min, np.array([2]))

    def test_decay_borders_wider_peak(self):
        hist = np.array([10, 19, 18, 15, 10, 5, 4, 3, 2, 10, 3])
        peaks = np.array([1])
        props = {"peak_heights": np.array([19])}
        decay, _max, _min = find_decay_borders(hist, peaks, props)
        np.testing.assert_equal(decay, np.array([19, 18, 15, 10, 5, 4, 3, 2]))
        np.testing.assert_equal(_max, np.array([19]))
        np.testing.assert_equal(_min, np.array([2]))

    def test_calc_lifetime(self):
        amp = 100
        tau = 35
        length = 125
        data = []
        for index in range(1, 126):
            data.extend([index for _ in range(1, int(amp * np.exp(-index / tau)) + 1)])
        tau = calc_lifetime(data)
        assert np.allclose([tau], [35], atol=0.5)

    def test_full_pipe(self):
        amp = 100
        tau = 35
        length = 125
        num_of_bins = 100
        data_of_single_bin = []
        bin_idx = []
        for index in range(1, length + 1):
            data_of_single_bin.extend([index for _ in range(1, int(amp * np.exp(-index / tau)) + 1)])
        for pixel_idx in range(num_of_bins):
            bin_idx.extend([pixel_idx for _ in range(len(data_of_single_bin))])
        data_of_single_bin  = num_of_bins * data_of_single_bin
        data_of_single_bin = np.array(data_of_single_bin)
        bin_idx = np.array(bin_idx)
        fl = FlimCalc(data_of_single_bin, bin_idx)
        fl.run()
        true_taus = np.ones((num_of_bins)) * tau
        bins = np.arange(num_of_bins)
        assert np.allclose(fl.hist_arrivals["since_laser"], true_taus, atol=0.5)
        assert np.allclose(fl.hist_arrivals["bin"], bins)
