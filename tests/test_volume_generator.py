"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
import numpy as np
import pandas as pd

from pysight.nd_hist_generator.volume_gen import VolumeGenerator


def gen_test_df(frame_num, end=1000000) -> pd.DataFrame:
    photons = np.arange(0, end, dtype=np.uint64)
    frames = np.linspace(0, end, num=frame_num, dtype=np.uint64,
                         endpoint=False)
    ones_frames = np.ones((1, int(len(photons) / len(frames))),
                          dtype=np.uint64)
    frames = (np.atleast_2d(frames).T @ ones_frames).ravel()
    assert len(frames) == len(photons)
    df = pd.DataFrame({'time_rel_frame': photons - frames,
                       'Frames': frames})
    df.set_index(['Frames'], drop=True, inplace=True)
    return df


class TestVolumeGenerator(TestCase):

    def test_many_frames(self):
        df = gen_test_df(1000)
        shape = (1000, 512, 512, 16)
        max_vols = int(300e6 / (np.prod(shape[1:]) * 8))
        volgen = VolumeGenerator(df, shape)
        volgen.create_frame_slices(create_slices=False)
        chunks = list(volgen.full_frame_chunks)
        self.assertEqual(len(chunks[0]), max_vols)

    def test_single_frame(self):
        df = gen_test_df(1000)
        shape = (10, 512, 512, 16)
        max_vols = 1
        volgen = VolumeGenerator(df, shape, MAX_BYTES_ALLOWED=100)
        volgen.create_frame_slices(create_slices=False)
        self.assertEqual(len(list(volgen.full_frame_chunks)[0]), max_vols)

    def test_vol_list(self):
        df = gen_test_df(10, end=100)
        shape = (10, 512, 512, 16)
        volgen = VolumeGenerator(df, shape)
        vol_times = volgen._VolumeGenerator__create_vol_list()
        self.assertSequenceEqual(vol_times, np.linspace(0, 100, 10, endpoint=False, dtype=np.int64).tolist())

    def test_standard_slice(self):
        df = gen_test_df(10, end=100)
        shape = (10, 512, 512, 16)
        volgen = VolumeGenerator(df, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 71), slice(80, 91)])

    def test_full_slice(self):
        df = gen_test_df(16, end=1600)
        shape = (16, 512, 512, 16)
        volgen = VolumeGenerator(df, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 701), slice(800, 1501)])

    def test_single_slice(self):
        df = gen_test_df(1, end=10)
        shape = (1, 512, 512, 16)
        volgen = VolumeGenerator(df, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 1)])

    def test_grouper(self):
        volgen = VolumeGenerator(pd.DataFrame(), (1,))
        volgen.frames = [10, 20, 30, 40]
        volgen.chunk_size = 3
        grouped = volgen._VolumeGenerator__grouper()
        self.assertSequenceEqual(list(grouped), [(10, 20, 30), (40, np.nan, np.nan)])

