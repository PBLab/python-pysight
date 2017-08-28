"""
__author__ = Hagai Hargil
"""

import unittest
import pandas as pd
import numpy as np
from pysight.gating_tools import GatedDetection


class TestGatedDetection(unittest.TestCase):
    df = pd.DataFrame([-1, 1, 5, 7, 9, 19, 25], columns=['time_rel_pulse'],
                      dtype=np.uint64)
    base_obj = GatedDetection(raw=df)

    def test_bins_bet_pulses(self):
        inst = []
        inst.append(self.base_obj)
        inst.append(GatedDetection(raw=self.df, reprate=160.3e6))
        inst.append(GatedDetection(raw=self.df, binwidth=100e-12))

        bins_bet_pulses = [16, 8, 125]
        for bins, obj in zip(bins_bet_pulses, inst):
            self.assertEqual(bins, obj.bins_bet_pulses)

    def test_validate_time(self):
        arr = np.array([[1], [5], [7], [9]])

        obj = self.base_obj
        obj._GatedDetection__validate_time_rel_pulse()
        self.assertSequenceEqual(arr.tolist(), obj.raw.values.tolist())

    def test_validate_with_boundaries(self):
        arr = np.array([[0], [5], [15]])
        df2 = pd.DataFrame([0, 5, 15, 16], columns=['time_rel_pulse'],
                           dtype=np.uint64)
        obj = GatedDetection(raw=df2)
        obj._GatedDetection__validate_time_rel_pulse()
        self.assertSequenceEqual(arr.tolist(), obj.raw.values.tolist())

    def test_discard_events(self):
        arr = np.array([1, 2, 3, 3, 3, 4, 4, 5, 6, 11, 13, 10, 15, 15], dtype=np.uint64)
        df = pd.DataFrame(arr, columns=['time_rel_pulse'], dtype=np.uint64)
        hist, _ = np.histogram(arr, bins=range(1, 17))
        obj = GatedDetection(raw=df)
        obj._GatedDetection__discard_events(hist=hist)
        self.assertSequenceEqual(obj.data.time_rel_pulse.tolist(), arr[:9].tolist())

    def test_discard_with_wrap(self):
        arr = np.array([1, 2, 3, 3, 4, 4, 11, 13, 14, 15, 15, 15, 1], dtype=np.uint64)
        df = pd.DataFrame(arr, columns=['time_rel_pulse'], dtype=np.uint64)
        hist, _ = np.histogram(arr, bins=16)
        obj = GatedDetection(raw=df)
        obj._GatedDetection__discard_events(hist=hist)
        ans_arr = [1, 2, 3, 3, 4, 4, 14, 15, 15, 15, 1]
        self.assertSequenceEqual(obj.data.time_rel_pulse.tolist(), ans_arr)
