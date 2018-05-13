"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
import numpy as np
import pandas as pd
from typing import Tuple

from pysight.nd_hist_generator.volume_gen import VolumeGenerator


def gen_test_df(frame_num=10, end=1000000) -> Tuple[pd.DataFrame, pd.Series]:
    photons = np.arange(0, end, dtype=np.uint64)
    frames = np.linspace(0, end, num=frame_num, dtype=np.uint64,
                         endpoint=False)
    ones_frames = np.ones((1, int(len(photons) / len(frames))),
                          dtype=np.uint64)
    frames_ser = pd.Series(frames)
    frames = (np.atleast_2d(frames).T @ ones_frames).ravel()
    assert len(frames) == len(photons)
    df = pd.DataFrame({'time_rel_frame': photons - frames,
                       'Frames': frames})
    df.set_index(['Frames'], drop=True, inplace=True)
    return df, frames_ser


class TestVolumeGenerator(TestCase):

    def test_many_frames(self):
        _, frames = gen_test_df(1000)
        shape = (1000, 512, 512, 16)
        max_vols = int(300e6 / (np.prod(shape[1:]) * 8))
        volgen = VolumeGenerator(frames, shape)
        volgen.create_frame_slices(create_slices=False)
        chunks = list(volgen.full_frame_chunks)
        self.assertEqual(len(chunks[0]), max_vols)

    def test_single_frame(self):
        _, frames = gen_test_df(1000)
        shape = (10, 512, 512, 16)
        max_vols = 1
        volgen = VolumeGenerator(frames, shape, MAX_BYTES_ALLOWED=100)
        volgen.create_frame_slices(create_slices=False)
        self.assertEqual(len(list(volgen.full_frame_chunks)[0]), max_vols)

    def test_standard_slice(self):
        _, frames = gen_test_df(10, end=100)
        shape = (10, 512, 512, 16)
        volgen = VolumeGenerator(frames, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 70), slice(80, 90)])

    def test_full_slice(self):
        _, frames = gen_test_df(16, end=1600)
        shape = (16, 512, 512, 16)
        volgen = VolumeGenerator(frames, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 700), slice(800, 1500)])

    def test_single_slice(self):
        _, frames = gen_test_df(1, end=10)
        shape = (1, 512, 512, 16)
        volgen = VolumeGenerator(frames, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 0)])

    def test_single_chunk(self):
        _, frames = gen_test_df(10, end=100)
        shape = (10, 800, 800, 200)
        volgen = VolumeGenerator(frames, shape)
        vol_times = volgen.create_frame_slices()
        self.assertSequenceEqual(list(vol_times), [slice(0, 10, None), slice(10, 20, None), slice(20, 30, None),
                                                   slice(30, 40, None), slice(40, 50, None), slice(50, 60, None),
                                                   slice(60, 70, None), slice(70, 80, None), slice(80, 90, None)])

    def test_grouper(self):
        volgen = VolumeGenerator(pd.Series(), (1,))
        volgen.frames = pd.Series([10, 20, 30, 40], dtype=np.uint64)
        volgen.frames_per_chunk = 3
        grouped = volgen._VolumeGenerator__grouper()
        self.assertSequenceEqual(list(grouped), [(10, 20, 30), (40, np.nan, np.nan)])

    def test_grouper_single_chunk(self):
        volgen = VolumeGenerator(pd.Series(), (1,))
        volgen.frames = pd.Series([10, 20, 30, 40], dtype=np.uint64)
        volgen.frames_per_chunk = 1
        grouped = volgen._VolumeGenerator__grouper()
        self.assertSequenceEqual(list(grouped), [(10,), (20,), (30,), (40,)])

