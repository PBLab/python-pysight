"""
__author__ = Hagai Hargil
"""


import unittest
from pysight.censor_tools import CensorCorrection
from pysight.censor_tools import CensoredVolume
import pandas as pd
from collections import deque, namedtuple
import numpy as np


class TestCensorTools(unittest.TestCase):
    length = 100
    df = pd.DataFrame([i for i in range(length)], columns=['abs_time'])
    tuptype = namedtuple('TestCensor', ('hist', 'edges'))
    cens = tuptype
    cens.hist = np.array([i for i in range(length)])
    cens.edges = np.array([1, 4, 8, 18, 25, length - 1])

    def test_min_time(self):
        censored = CensorCorrection(df=self.df, reprate=80e6,
                                    deque_of_vols=deque([self.cens]),
                                    binwidth=800e-12, offset=9)
        self.assertEqual(censored.start_time, 0)

    def test_max_time(self):
        censored = CensorCorrection(df=self.df, reprate=80e6,
                                    deque_of_vols=deque([self.cens]),
                                    binwidth=800e-12, offset=9)
        self.assertEqual(censored.end_time, self.length - 1 + 16)

    def test_laser_pulses(self):
        censored = CensorCorrection(df=self.df, reprate=80e6,
                                    deque_of_vols=deque([self.cens]),
                                    binwidth=800e-12, offset=9)
        real_range = np.arange(start=0+censored.offset, stop=self.length-1+16, step=16, dtype=np.uint64)
        self.assertListEqual(list(real_range), list(censored.laser_pulses))

    def test_bincount(self):
        censored = CensorCorrection(df=self.df, reprate=80e6,
                                    deque_of_vols=deque([self.cens]),
                                    binwidth=800e-12, offset=9)
        real_result = np.bincount(np.histogram(self.cens.hist, bins=np.arange(start=0+censored.offset,
                                                                              stop=self.length-1+16,
                                                                              step=16, dtype=np.uint64))[0])
        self.assertListEqual(list(real_result), list(censored.gen_bincount()))

    def test_sort_photons(self):
        # censored = CensorCorrection(df=self.df, reprate=80e6,
        #                             deque_of_vols=deque([self.cens]),
        #                             binwidth=800e-12, offset=9)
        #
        # real_pulses = np.arange(start=0+censored.offset, stop=99+16, step=16, dtype=np.uint64)
        # sorted_indices = np.searchsorted(np.arange(start=9, stop=99+16, step=16, dtype=np.uint64), [i for i in range(100)])
        pass
