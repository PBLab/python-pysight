from collections import namedtuple
from pprint import pprint

import pytest

from pysight.nd_hist_generator.frame_chunk import *
from pysight.nd_hist_generator.movie import *


def gen_data_df(frame_num=10, line_num=100, end=1000, channels=2):
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
    if channels == 2:
        channel[len(channel) // 2 :] = 2
    lines = np.linspace(0, end, num=line_num, endpoint=False, dtype=np.uint64)
    x_pix = int(len(photons) / len(lines))
    ones_lines = np.ones((1, int(len(photons) / len(lines))), dtype=np.uint64)
    frames = np.linspace(0, end, num=frame_num, dtype=np.uint64, endpoint=False)
    frames_ser = pd.Series(frames, index=frames)
    ones_frames = np.ones((1, int(len(photons) / len(frames))), dtype=np.uint64)
    lines = (np.atleast_2d(lines).T @ ones_lines).ravel()
    frames = (np.atleast_2d(frames).T @ ones_frames).ravel()
    assert len(lines) == len(frames) == len(photons)

    df = pd.DataFrame(
        {
            "abs_time": photons,
            "time_rel_line": photons - lines,
            "Lines": lines,
            "Frames": frames,
            "Channel": channel,
        }
    )
    df.set_index(["Channel", "Frames", "Lines"], drop=True, inplace=True)
    y_pix = x_pix
    lines_to_return = pd.Series(
        np.unique(lines), index=np.repeat(frames_ser, line_num // frame_num)
    )

    return df, frames_ser, lines_to_return, x_pix, y_pix


class TestFrameChunk:
    df, frames, lines, x, y = gen_data_df()
    movie_single = Movie(
        df,
        lines,
        outputs={"memory": True},
        line_delta=int(lines.diff().mean()),
        fill_frac=100.0,
        bidir=True,
        data_shape=(len(frames), x, y),
        frame_slices=(slice(frame) for frame in frames),
        frames=frames,
        frames_per_chunk=10,
    )
    df_dict = {
        1: df.xs(key=(1, 100), level=("Channel", "Frames"), drop_level=False),
        2: df.xs(key=(2, 100), level=("Channel", "Frames"), drop_level=False),
    }
    chunk_single = FrameChunk(
        movie=movie_single,
        df_dict=df_dict,
        frames_per_chunk=10,
        frames=frames,
        lines=lines,
    )

    movie_multi = Movie(
        df,
        lines=lines,
        outputs={"memory": True},
        line_delta=int(lines.diff().mean()),
        fill_frac=100.0,
        bidir=True,
        data_shape=(len(frames), x, y),
        frames=frames,
        frame_slices=(slice(frame) for frame in frames),
        frames_per_chunk=4,
    )
    sl = pd.IndexSlice[slice(1), slice(100, 400)]
    chunk_multi = FrameChunk(
        movie=movie_multi,
        df_dict={1: df.loc[sl, :]},
        frames_per_chunk=4,
        frames=frames,
        lines=lines,
    )

    def test_frame_edges_single_chunk(self):
        fr = self.chunk_single._FrameChunk__create_frame_edges()
        np.testing.assert_equal(
            fr,
            np.array(
                [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 901], dtype=np.uint64
            ),
        )

    def test_frame_edges_multiple_frames(self):
        sl = pd.IndexSlice[slice(1), slice(100, 400)]
        movie = Movie(
            self.df,
            self.lines,
            outputs={"memory": True},
            line_delta=int(self.lines.diff().mean()),
            fill_frac=100.0,
            bidir=True,
            data_shape=(len(self.frames), self.x, self.y),
            frames=self.frames,
            frame_slices=(slice(frame) for frame in self.frames),
            frames_per_chunk=4,
        )
        chunk_multi = FrameChunk(
            movie=movie,
            df_dict={1: self.df.loc[sl, :]},
            frames=pd.Series([100, 200, 300, 400], dtype=np.uint64),
            lines=pd.Series([10]),
            frames_per_chunk=4,
        )
        fr = chunk_multi._FrameChunk__create_frame_edges()
        np.testing.assert_equal(fr, np.array([100, 200, 300, 400, 401]))

    def test_line_edges_single_chunk(self):
        li = self.chunk_single._FrameChunk__create_line_edges()
        lines = np.arange(0, 1010, 10)
        np.testing.assert_equal(li, lines)

    def test_line_edges_downsample(self):
        li = self.chunk_single._FrameChunk__create_line_edges(downsample=1)
        lines = np.arange(0, 1010, 10)
        np.testing.assert_equal(li, lines)

    def test_line_edges_downsample_2(self):
        li = self.chunk_single._FrameChunk__create_line_edges(downsample=2)
        lines = np.arange(0, 1020, 20)

    def test_line_edges_downsample_4(self):
        li = self.chunk_single._FrameChunk__create_line_edges(downsample=4)
        lines = np.arange(0, 1020, 40)

    def test_line_edges_multi_chunk(self):
        sl = pd.IndexSlice[slice(1), slice(100, 400)]
        movie = Movie(
            self.df,
            self.lines,
            outputs={"memory": True},
            line_delta=int(self.lines.diff().mean()),
            fill_frac=100.0,
            bidir=True,
            data_shape=(len(self.frames), self.x, self.y),
            frames=self.frames,
            frame_slices=(slice(frame) for frame in self.frames),
            frames_per_chunk=4,
        )
        chunk_multi = FrameChunk(
            movie=movie,
            df_dict={1: self.df.loc[sl, :]},
            frames=self.frames,
            lines=self.lines.loc[slice(100, 400)],
            frames_per_chunk=4,
        )
        li = chunk_multi._FrameChunk__create_line_edges()
        lines = np.arange(100, 510, 10)
        np.testing.assert_equal(li, lines)

    def test_col_edges_single_frame(self):
        cr = self.chunk_single._FrameChunk__create_col_edges()
        cols = np.arange(11)
        np.testing.assert_equal(cr, cols)

    def test_col_edges_single_frame_downsample(self):
        cr = self.chunk_single._FrameChunk__create_col_edges(downsample=1)
        cols = np.arange(11)
        np.testing.assert_equal(cr, cols)

    def test_col_edges_single_frame_downsample_2(self):
        cr = self.chunk_single._FrameChunk__create_col_edges(downsample=2)
        cols = np.arange(0, 11, 2)
        np.testing.assert_equal(cr, cols)

    def test_col_edges_single_frame_downsample_5(self):
        cr = self.chunk_single._FrameChunk__create_col_edges(downsample=5)
        cols = np.arange(0, 11, 5)
        np.testing.assert_equal(cr, cols)

    def test_col_edges_multi_frame(self):
        cr = self.chunk_multi._FrameChunk__create_col_edges()
        cols = np.arange(11)
        np.testing.assert_equal(cr, cols)

    def test_linspace_along_sine_1_pix_z(self):
        sine = self.chunk_single._FrameChunk__linspace_along_sine()
        np.testing.assert_almost_equal(
            np.array([-0.999_995_03, 0.999_995_01]), sine.tolist()
        )

    def test_linspace_along_sine_100_pix_z(self):
        movie_for_sine = Movie(
            self.df,
            self.lines,
            data_shape=(1, 512, 512, 100),
            frame_slices=(1,),
            frames=self.frames,
        )
        chunk = FrameChunk(
            movie_for_sine,
            self.df_dict,
            frames_per_chunk=1,
            frames=self.frames,
            lines=self.lines,
        )
        sin = chunk._FrameChunk__linspace_along_sine()
        true = np.array(
            [
                -9.999_950_30e-01,
                -9.800_004_36e-01,
                -9.600_004_08e-01,
                -9.400_011_49e-01,
                -9.200_012_49e-01,
                -9.000_018_13e-01,
                -8.800_007_00e-01,
                -8.600_020_42e-01,
                -8.400_002_45e-01,
                -8.200_015_19e-01,
                -8.000_050_73e-01,
                -7.800_049_16e-01,
                -7.600_003_63e-01,
                -7.400_022_30e-01,
                -7.200_009_40e-01,
                -7.000_044_11e-01,
                -6.800_007_61e-01,
                -6.600_036_90e-01,
                -6.400_041_55e-01,
                -6.200_007_61e-01,
                -6.000_020_52e-01,
                -5.800_040_59e-01,
                -5.600_065_23e-01,
                -5.400_055_25e-01,
                -5.200_023_25e-01,
                -5.000_042_41e-01,
                -4.799_990_88e-01,
                -4.599_986_46e-01,
                -4.400_044_90e-01,
                -4.199_985_05e-01,
                -3.999_970_92e-01,
                -3.799_975_74e-01,
                -3.599_960_63e-01,
                -3.399_969_68e-01,
                -3.200_039_66e-01,
                -3.000_009_72e-01,
                -2.799_995_81e-01,
                -2.600_014_20e-01,
                -2.399_978_86e-01,
                -2.199_992_22e-01,
                -2.000_056_36e-01,
                -1.800_071_04e-01,
                -1.600_029_81e-01,
                -1.400_022_36e-01,
                -1.200_037_63e-01,
                -1.000_062_21e-01,
                -8.000_806_72e-02,
                -6.000_760_10e-02,
                -4.000_299_65e-02,
                -2.000_233_92e-02,
                -3.673_215_39e-06,
                1.999_499_42e-02,
                3.999_565_59e-02,
                6.000_026_79e-02,
                8.000_074_44e-02,
                9.999_891_11e-02,
                1.199_964_70e-01,
                1.399_949_62e-01,
                1.599_957_30e-01,
                1.799_998_78e-01,
                1.999_984_38e-01,
                2.200_018_11e-01,
                2.400_004_62e-01,
                2.599_943_26e-01,
                2.800_021_29e-01,
                2.999_939_64e-01,
                3.199_970_06e-01,
                3.399_994_63e-01,
                3.599_985_38e-01,
                3.800_000_28e-01,
                3.999_995_24e-01,
                4.200_009_14e-01,
                4.399_978_93e-01,
                4.600_010_02e-01,
                4.800_014_16e-01,
                4.999_978_79e-01,
                5.199_960_50e-01,
                5.399_993_41e-01,
                5.600_004_36e-01,
                5.799_980_74e-01,
                5.999_961_74e-01,
                6.199_949_97e-01,
                6.399_985_10e-01,
                6.599_981_71e-01,
                6.799_953_75e-01,
                6.999_991_65e-01,
                7.200_027_81e-01,
                7.399_972_88e-01,
                7.599_955_88e-01,
                7.800_003_19e-01,
                7.999_946_65e-01,
                8.199_973_14e-01,
                8.400_016_84e-01,
                8.599_982_93e-01,
                8.799_972_10e-01,
                8.999_986_10e-01,
                9.199_983_70e-01,
                9.399_986_43e-01,
                9.599_983_51e-01,
                9.799_989_74e-01,
                9.999_950_07e-01,
            ]
        )
        np.testing.assert_almost_equal(sin, true)


def test_calc_lifetime():
    amp = 100
    tau = 35
    length = 125
    data = []
    for index in range(1, length + 1):
        data.extend([index for _ in range(1, int(amp * np.exp(-index / tau)) + 1)])
    comp_tau = calc_lifetime(data)
    assert np.allclose([comp_tau], [35], atol=1)

