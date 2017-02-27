"""
__author__ = Hagai Hargil
"""
from unittest import TestCase
import pandas as pd


class TestFrame(TestCase):
    """
    Tests for the Frame class
    """
    list_of_file_names = [r'..\TAG ON start channel, galvo on stop 2 - gain 480.lst',
                          r'..\fixed sample - XY image - 500 gain - 1 second acquisition.lst'
                          r'..\live mouse  100 um deep with 62p TAG010.lst'
                          r'..\multiscaler_check_code_2_channels_0-1_sec.lst']

    data = pd.read_hdf('df_for_test.h5')

    def test_first_line_time(self):
        self.fail()

    def test_last_line_time(self):
        self.fail()

    def test_first_event_time(self):
        self.fail()

    def test_max_delta_of_lines(self):
        self.fail()

    def test_create_hist_edges(self):
        self.fail()

    def test_create_hist(self):
        self.fail()

    def test_display(self):
        self.fail()
