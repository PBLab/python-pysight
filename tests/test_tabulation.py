import unittest
import numpy as np

from pysight.ascii_list_file_parser.tabulation import *


class TestTabulation(unittest.TestCase):

    tab = Tabulate(dict_of_inputs={}, data=np.array([]),
                   dict_of_slices_hex={})
    def test_conversion_hex_to_bits(self):
        diction = \
            {
                '0': '0000',
                '1': '0001',
                '2': '0010',
                '3': '0011',
                '4': '0100',
                '5': '0101',
                '6': '0110',
                '7': '0111',
                '8': '1000',
                '9': '1001',
                'a': '1010',
                'b': '1011',
                'c': '1100',
                'd': '1101',
                'e': '1110',
                'f': '1111',
            }
        self.assertEqual(diction, self.tab.hex_to_bin_dict())
