"""
__author__ = Hagai Hargil
"""


import unittest
from pysight.censor_tools import CensorCorrection
from pysight.censor_tools import CensoredVolume
from pysight.movie_tools import Movie, Volume
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

    dict_of_data = {
        'PMT1': pd.DataFrame([[1, 2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 9, 10, 11]],
                             columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep',
                                      'time_rel_pulse', 'bins_x', 'bins_y'], dtype=np.uint64),
        'Lines': pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                             columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep']),
        'Laser': pd.DataFrame([[0, 2, 3, 2], [3, 6, 7, 6]],
                             columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep'])
    }
    movie = Movie(data=dict_of_data['PMT1'])


    def test_allocate_empty(self):
        censored = CensoredVolume(df=self.df, vol=Volume(data=self.dict_of_data['PMT1']),
                                  laser_pulses=self.dict_of_data['Laser'].values,
                                  offset=0)
        empty_hist = np.zeros((16,), dtype='uint8')

        self.assertTrue(np.all(empty_hist == censored._CensoredVolume__allocate_empty_to_bins()))

    def test_allocate_some_photons(self):
        censored = CensoredVolume(df=self.df, vol=Volume(data=self.dict_of_data['PMT1']),
                                  laser_pulses=self.dict_of_data['Laser'].values,
                                  offset=0)
        photons = self.dict_of_data['PMT1']
        photons.set_index(keys=['bins_x', 'bins_y'], drop=True, inplace=True)
        col = 7
        res_hist, _ = np.histogram(np.array([5]),
                                   bins=np.arange(0, censored.bins_bet_pulses+1, dtype=np.uint64))
        self.assertTrue(np.all(res_hist == censored._CensoredVolume__allocate_photons_to_bins(col, photons)))

    def test_allocate_no_photons(self):
        censored = CensoredVolume(df=self.df, vol=Volume(data=self.dict_of_data['PMT1']),
                                  laser_pulses=self.dict_of_data['Laser'].values,
                                  offset=0)
        photons = self.dict_of_data['PMT1']
        photons.set_index(keys=['bins_x', 'bins_y'], drop=True, inplace=True)
        col = 9
        res_hist, _ = np.histogram(np.array([]),
                                   bins=np.arange(0, censored.bins_bet_pulses + 1, dtype=np.uint64))
        self.assertTrue(np.all(res_hist == censored._CensoredVolume__allocate_photons_to_bins(col, photons)))

    def test_append_laser(self):
        censored = CensorCorrection(raw=self.dict_of_data,
                                    movie=self.movie,
                                    all_laser_pulses=self.dict_of_data['Laser'])

        censored.append_laser_line()
        res = pd.DataFrame([[0, 2, 3, 2], [3, 6, 7, 6], [19, 0, 7, 22]],
                           columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep'])
        self.assertTrue(censored.raw['Laser'].equals(res))
