"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
from collections import namedtuple
from pprint import pprint

from pysight.nd_hist_generator.frame_chunk import *
from pysight.nd_hist_generator.movie_tools import *


def gen_data_df():
    """
    Mock data for tests.
    Returns:
        df - The full DataFrame
        lines only
        x pixels
        y pixels
    """
    photons = np.arange(0, 1000, dtype=np.uint64)
    channel = np.ones_like(photons)
    channel[channel.shape[0] // 2:] = 2
    lines = np.arange(0, 1000, step=10, dtype=np.uint64)
    x_pix = int(len(photons) / len(lines))
    ones_lines = np.ones((1, int(len(photons) / len(lines))),
                         dtype=np.uint64)
    frames = np.arange(0, 1000, step=100, dtype=np.uint64)
    ones_frames = np.ones((1, int(len(photons) / len(frames))),
                          dtype=np.uint64)
    lines = (np.atleast_2d(lines).T @ ones_lines).ravel()
    frames = (np.atleast_2d(frames).T @ ones_frames).ravel()
    assert len(lines) == len(frames) == len(photons)

    df = pd.DataFrame({'time_rel_line': photons - lines,
                       'time_rel_frames': photons - frames,
                       'Lines': lines, 'Frames': frames,
                       'Channel': channel})
    df.set_index(['Channel', 'Frames', 'Lines'], drop=True, inplace=True)
    y_pix = x_pix
    return df, pd.Series(np.unique(lines)), x_pix, y_pix, frames


class TestFrameChunk(TestCase):
    df, lines, x, y, frames = gen_data_df()
    movie_single = Movie(df, lines,
                  outputs={'memory': True}, line_delta=int(lines.diff().mean()),
                  fill_frac=100., bidir=True, data_shape=(len(frames), x, y),
                  frames=(slice(frame) for frame in frames))
    df_dict = {1: df.xs(key=(1, 100), level=('Channel', 'Frames'),
                        drop_level=False),
               2: df.xs(key=(2, 200), level=('Channel', 'Frames'),
                        drop_level=False)}
    chunk_single = FrameChunk(movie=movie_single, df_dict=df_dict, frames_per_chunk=1)

    movie_multi = Movie(df, lines, outputs={'memory': True},
                        line_delta=int(lines.diff().mean()), fill_frac=100., bidir=True,
                        data_shape=(len(frames), x, y),
                        frames=(slice(frame) for frame in frames))
    sl = pd.IndexSlice[slice(1), slice(100, 400)]
    chunk_multi = FrameChunk(movie=movie_multi, df_dict={1: df.loc[sl, :]}, frames_per_chunk=4)

    def test_frame_edges_single_frame(self):
        fr = self.chunk_single._FrameChunk__create_frame_and_line_edges(1)
        self.assertSequenceEqual(fr.tolist(), np.arange(100, 210, 10, dtype=np.uint64).tolist())

    def test_frame_edges_multiple_frames(self):
        fr = self.chunk_multi._FrameChunk__create_frame_and_line_edges(1)
        arr = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 200, 210,
                        220, 230, 240, 250, 260, 270, 280, 290, 300, 300, 310, 320, 330,
                        340, 350, 360, 370, 380, 390, 400, 400, 410, 420, 430, 440, 450,
                        460, 470, 480, 490, 500], dtype=np.uint64)
        np.testing.assert_equal(fr, arr)

    def test_col_edges_single_frame(self):
        cr = self.chunk_single._FrameChunk__create_col_edges(1)
        cols = np.arange(11)
        np.testing.assert_equal(cr, cols)

    def test_col_edges_multi_frame(self):
        cr = self.chunk_multi._FrameChunk__create_col_edges(1)
        cols = np.arange(11)
        self.assertSequenceEqual(cr.tolist(), cols.tolist())

    def test_linspace_along_sine_1_pix_z(self):
        sine = self.chunk_single._FrameChunk__linspace_along_sine()
        np.testing.assert_almost_equal(np.array([-0.99999503, 0.99999501]),
                                       sine.tolist())

    def test_linspace_along_sine_100_pix_z(self):
        movie_for_sine = Movie(self.df, self.lines, data_shape=(1, 512, 512, 100),
                               frames=(1,))
        chunk = FrameChunk(movie_for_sine, self.df_dict)
        sin = chunk._FrameChunk__linspace_along_sine()
        true = np.array([-9.99995030e-01, -9.80000436e-01, -9.60000408e-01, -9.40001149e-01,
                         -9.20001249e-01, -9.00001813e-01, -8.80000700e-01, -8.60002042e-01,
                         -8.40000245e-01, -8.20001519e-01, -8.00005073e-01, -7.80004916e-01,
                         -7.60000363e-01, -7.40002230e-01, -7.20000940e-01, -7.00004411e-01,
                         -6.80000761e-01, -6.60003690e-01, -6.40004155e-01, -6.20000761e-01,
                         -6.00002052e-01, -5.80004059e-01, -5.60006523e-01, -5.40005525e-01,
                         -5.20002325e-01, -5.00004241e-01, -4.79999088e-01, -4.59998646e-01,
                         -4.40004490e-01, -4.19998505e-01, -3.99997092e-01, -3.79997574e-01,
                         -3.59996063e-01, -3.39996968e-01, -3.20003966e-01, -3.00000972e-01,
                         -2.79999581e-01, -2.60001420e-01, -2.39997886e-01, -2.19999222e-01,
                         -2.00005636e-01, -1.80007104e-01, -1.60002981e-01, -1.40002236e-01,
                         -1.20003763e-01, -1.00006221e-01, -8.00080672e-02, -6.00076010e-02,
                         -4.00029965e-02, -2.00023392e-02, -3.67321539e-06,  1.99949942e-02,
                          3.99956559e-02,  6.00002679e-02,  8.00007444e-02,  9.99989111e-02,
                          1.19996470e-01,  1.39994962e-01,  1.59995730e-01,  1.79999878e-01,
                          1.99998438e-01,  2.20001811e-01,  2.40000462e-01,  2.59994326e-01,
                          2.80002129e-01,  2.99993964e-01,  3.19997006e-01,  3.39999463e-01,
                          3.59998538e-01,  3.80000028e-01,  3.99999524e-01,  4.20000914e-01,
                          4.39997893e-01,  4.60001002e-01,  4.80001416e-01,  4.99997879e-01,
                          5.19996050e-01,  5.39999341e-01,  5.60000436e-01,  5.79998074e-01,
                          5.99996174e-01,  6.19994997e-01,  6.39998510e-01,  6.59998171e-01,
                          6.79995375e-01,  6.99999165e-01,  7.20002781e-01,  7.39997288e-01,
                          7.59995588e-01,  7.80000319e-01,  7.99994665e-01,  8.19997314e-01,
                          8.40001684e-01,  8.59998293e-01,  8.79997210e-01,  8.99998610e-01,
                          9.19998370e-01,  9.39998643e-01,  9.59998351e-01,  9.79998974e-01,
                          9.99995007e-01])
        np.testing.assert_almost_equal(sin, true)
