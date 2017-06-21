"""
__author__ = Hagai Hargil
"""


import unittest
from pysight.censor_tools import CensorCorrection
from pysight.censor_tools import CensoredVolume
from pysight.movie_tools import Movie
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
        'PMT1': pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                             columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep']),
        'Lines': pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                             columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep']),
        'Laser': pd.DataFrame([[0, 2, 3, 2], [3, 6, 7, 6]],
                             columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep'])
    }
    movie = Movie(data=dict_of_data['PMT1'])

    def test_train_dataset(self):
        censored = CensorCorrection(raw=self.dict_of_data,
                                    movie=self.cens,
                                    all_laser_pulses=self.dict_of_data['Laser'])


    def test_append_laser(self):
        censored = CensorCorrection(raw=self.dict_of_data,
                                    movie=self.movie,
                                    all_laser_pulses=self.dict_of_data['Laser'])

        censored.append_laser_line()
        res = pd.DataFrame([[0, 2, 3, 2], [3, 6, 7, 6], [19, 0, 7, 22]],
                           columns=['abs_time', 'edge', 'sweep', 'time_rel_sweep'])
        self.assertTrue(censored.raw['Laser'].equals(res))
