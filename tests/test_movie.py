"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
from pysight.nd_hist_generator.movie_tools import *
import pandas as pd
import numpy as np


def gen_data_df():
    """
    Mock data for tests.
    Returns:
        df - The full DataFrame
        lines only
        x pixels
        y pixels
    """
    photons = np.arange(0, 1000, dtype=np.int64)
    channel = np.ones_like(photons)
    lines = np.arange(0, 1000, step=10, dtype=np.int64)
    x_pix = int(len(photons) / len(lines))
    ones_lines = np.ones((1, int(len(photons) / len(lines))),
                         dtype=np.int64)
    frames = np.arange(0, 1000, step=100, dtype=np.int64)
    ones_frames = np.ones((1, int(len(photons) / len(frames))),
                          dtype=np.int64)
    lines = (np.atleast_2d(lines).T @ ones_lines).ravel()
    frames = (np.atleast_2d(frames).T @ ones_frames).ravel()
    assert len(lines) == len(frames) == len(photons)

    df = pd.DataFrame({'time_rel_line': photons - lines,
                       'time_rel_frames': photons - frames,
                       'Lines': lines, 'Frames': frames,
                       'Channel': channel})
    df.set_index(['Channel', 'Frames', 'Lines'], drop=True, inplace=True)
    y_pix = x_pix
    return df, pd.Series(np.unique(lines)), x_pix, y_pix


class TestMovies(TestCase):
    data, lines, x_pix, y_pix = gen_data_df()
    movie = Movie(data, lines, x_pixels=x_pix, y_pixels=y_pix,
                 outputs={'memory': True}, line_delta=int(lines.diff().mean()),
                 fill_frac=100., bidir=True)
    movie.run()

    def test_all_pipeline_basic(self):
        self.assertTrue(np.all(self.movie.stack[1].ravel() == np.ones((1000,), dtype=np.uint8)))

    def test_baseline_outputs(self):
        during, end = self.movie._Movie__determine_outputs()
        self.assertTrue(len(during) == 1)
        self.assertTrue(len(end) == 1)
        during, end = str(during), str(end)
        self.assertTrue('create_memory_output' in during)
        self.assertTrue('convert_deque_to_arr' in end)

