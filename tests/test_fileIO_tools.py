"""
__author__ = Hagai Hargil
"""
import unittest
from os import sep


class TestFileIOTools(unittest.TestCase):
    """
        Tests for new multiscaler readout functions
        """
    list_of_file_names = ['tests_data' + sep + 'data_for_tests.lst']

    list_of_real_start_loc = [1749]
    list_of_real_time_patch = ['32']
    list_of_real_range = [80000000 * 2 ** 4]

    def test_check_range_extraction(self):
        from pysight.fileIO_tools import get_range

        list_of_returned_range = []
        for fname in self.list_of_file_names:
            list_of_returned_range.append(get_range(fname))

        self.assertEqual(self.list_of_real_range, list_of_returned_range)

    def test_check_time_patch_extraction(self):
        from pysight.fileIO_tools import get_timepatch

        list_of_returned_time_patch = []
        for fname in self.list_of_file_names:
            list_of_returned_time_patch.append(get_timepatch(fname))

        self.assertEqual(self.list_of_real_time_patch, list_of_returned_time_patch)

    def test_check_start_of_data_value(self):
        from pysight.fileIO_tools import get_start_pos

        list_of_returned_locs = []
        for fname in self.list_of_file_names:
            list_of_returned_locs.append(get_start_pos(fname))

        self.assertEqual(self.list_of_real_start_loc, list_of_returned_locs)
