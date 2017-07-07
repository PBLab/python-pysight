import unittest
import pandas as pd
import numpy as np
from os import sep
from pysight.lst_tools import Analysis
from pysight.fileIO_tools import FileIO
from pysight import timepatch_switch


class TestLstTools(unittest.TestCase):
    length = 100
    df = pd.DataFrame([i for i in range(length)], columns=['abs_time'])
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
    cur_file = FileIO(filename='tests_data' + sep + 'data_for_tests_1.lst')
    cur_file.run()
    dict_of_slices_hex = timepatch_switch.ChoiceManagerHex().process(cur_file.timepatch)
    analyzed = Analysis(dict_of_inputs=cur_file.dict_of_input_channels, data=cur_file.data,
                        dict_of_slices_hex=dict_of_slices_hex, dict_of_slices_bin=None)

    def test_censor_not_needed(self):
        a = np.array([10, 9, 8, 5])
        self.assertTrue(not self.analyzed._Analysis__requires_censoring(a))

    def test_censor_needed_1(self):
        a = np.array([10, 9, 0, 5])
        self.assertTrue(self.analyzed._Analysis__requires_censoring(a))

    def test_censor_needed_2(self):
        a = np.array([10, 9, 0, 7, 5, 0, 2])
        self.assertTrue(self.analyzed._Analysis__requires_censoring(a))

    def test_censor_needed_3(self):
        a = np.array([10, 9, 0, 7, 5, 0, 2, 5])
        self.assertTrue(self.analyzed._Analysis__requires_censoring(a))

    def test_censor_needed_4(self):
        a = np.array([10, 9, 0, 7, 5, 0, 2, 0])
        self.assertTrue(self.analyzed._Analysis__requires_censoring(a))

    def test_laser_interpolation(self):
        pass


if __name__ == '__main__':
    unittest.main()
