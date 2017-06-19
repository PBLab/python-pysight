"""
__author__ = Hagai Hargil
"""

from unittest import TestCase
from pysight.validation_tools import *


class TestFrame(TestCase):
    """
    Tests for the validation functions
    """
    import pandas as pd
    import numpy as np

    def test_last_event_empty_dict(self):
        dict_of_data = dict()
        lines_per_frame = 1
        with self.assertRaises(ValueError):
            calc_last_event_time(dict_of_data, lines_per_frame)

    def test_last_event_wrong_lines(self):
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time'])}
        lines_per_frame = 0
        with self.assertRaises(ValueError):
            calc_last_event_time(dict_of_data, lines_per_frame)

    def test_last_event_only_pmt(self):
        dict_of_data = {'PMT1': pd.DataFrame([1, 5, 3], columns=['abs_time'])}
        lines_per_frame = 1
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         5)

    def test_last_event_with_frames(self):
        frame_data = pd.DataFrame([0, 100], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Frames': frame_data,
                        'Lines': [1, 2, 3]}
        lines_per_frame = 1
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         200)

    def test_last_event_with_single_frame(self):
        frame_data = pd.DataFrame([100], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Frames': frame_data,
                        'Lines': [1, 2, 3]}
        lines_per_frame = 1
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         200)

    def test_last_event_with_lines_less_than_needed_single_frame(self):
        line_data = pd.DataFrame([0, 10, 20], columns=['abs_time'])
        lines_per_frame = 5
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         50)

    def test_last_event_with_lines_less_than_needed_more_frames(self):
        line_data = pd.DataFrame([0, 10, 20, 30, 40], columns=['abs_time'])
        lines_per_frame = 3
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         60)

    def test_last_event_with_lines_just_like_needed(self):
        line_data = pd.DataFrame([0, 10, 20], columns=['abs_time'])
        lines_per_frame = 3
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         30)

    def test_last_event_with_lines_more_than_needed(self):
        line_data = pd.DataFrame(np.arange(0, 100, 10), columns=['abs_time'])
        lines_per_frame = 3
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        self.assertEqual(calc_last_event_time(dict_of_data, lines_per_frame),
                         120)
