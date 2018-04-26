"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
from collections import namedtuple

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
    movie = Movie(df, lines,
                  outputs={'memory': True}, line_delta=int(lines.diff().mean()),
                  fill_frac=100., bidir=True, data_shape=(len(frames), x, y),
                  frames=(slice(frame) for frame in frames), frames_per_chunk=1,)
    df_dict = {1: df.xs(key=(1, 100), level=('Channel', 'Frames'),
                        drop_level=False),
               2: df.xs(key=(2, 200), level=('Channel', 'Frames'),
                        drop_level=False)}
    chunk = FrameChunk(movie=movie, df_dict=df_dict)

    def test_single_frame_edges(self):
        fr = self.chunk._FrameChunk__create_frame_edges(1)
        self.assertSequenceEqual(fr.tolist(), np.array([100, 101], dtype=np.uint64).tolist())

    def test_multiple_frame_edges(self):
        sl = pd.IndexSlice[slice(1), slice(100, 400)]
        movie = Movie(self.df, self.lines, outputs={'memory': True},
                      line_delta=int(self.lines.diff().mean()), fill_frac=100., bidir=True,
                      data_shape=(len(self.frames), self.x, self.y),
                      frames=(slice(frame) for frame in self.frames), frames_per_chunk=4, )
        chunk_multi = FrameChunk(movie=movie, df_dict={1: self.df.loc[sl, :]})
        fr = chunk_multi._FrameChunk__create_frame_edges(1)
        self.assertSequenceEqual(fr.tolist(), np.array([100, 200, 300, 400, 401]).tolist())

    def test_line_edges(self):
        lr = self.chunk._FrameChunk__create_line_edges(1)
        lines = np.arange(100, 200, 10)
        self.assertSequenceEqual(lr.tolist(), lines.tolist())
