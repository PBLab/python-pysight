"""
__author__ = Hagai Hargil
"""


import unittest
from pysight.nd_hist_generator.censor_correction import CensorCorrection
from pysight.nd_hist_generator.censor_correction import CensoredVolume
from pysight.nd_hist_generator.movie import Movie
import pandas as pd
from collections import namedtuple
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
    dict_of_data['PMT1'].time_rel_pulse = dict_of_data['PMT1'].time_rel_pulse.astype(np.uint8)
    movie = Movie(data=dict_of_data['PMT1'], lines=dict_of_data['Lines'].abs_time, frames=pd.Series([]),
                  frame_slices=())

    # def test_allocate_some_photons(self):
    #     censored = CensoredVolume(df=self.df, vol=Volume(data=self.dict_of_data['PMT1'],
    #                                                      lines=self.dict_of_data['Lines'].abs_time),
    #                               offset=0)
    #     photons = self.dict_of_data['PMT1']
    #     photons.set_index(keys=['bins_x', 'bins_y'], drop=True, inplace=True)
    #     idx_list = np.array([7])
    #     result = np.array([0], dtype=object)
    #     res_hist = np.histogram(np.array([5], dtype=object),
    #                             bins=np.arange(0, censored.bins_bet_pulses+1, dtype=np.uint8))[0]
    #     result[0] = res_hist
    #     returned = censored._CensoredVolume__allocate_photons_to_bins(idx_list, photons)
    #     self.assertTrue(np.all(result[0] == returned[0]))

    def test_append_laser(self):
        censored = CensorCorrection(raw=self.dict_of_data,
                                    movie=self.movie,
                                    all_laser_pulses=self.dict_of_data['Laser'],
                                    data=pd.DataFrame())

        censored.append_laser_line()
        res = pd.DataFrame([[0, 2, 3, 2], [3, 6, 7, 6], [19, 0, 7, 22]],
                           columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep'])
        self.assertTrue(censored.raw['Laser'].equals(res))
