from unittest import TestCase

from pysight.binary_list_file_parser.binary_parser import *

import numpy as np


class BinaryTest(TestCase):
    data = np.array([3, 7, 15, 16, 8])
    bindata = BinaryDataParser(data, timepatch='5b')

    def test_chan_standard(self):
        chan = self.bindata._BinaryDataParser__get_channel()
        np.testing.assert_array_equal(chan, np.array([3, 7, 7, 0, 0],
                                                     dtype=np.uint8))

    def test_edge_standard(self):
        edge = self.bindata._BinaryDataParser__get_edge()
        np.testing.assert_array_equal(edge, np.array([0, 0, 1, 0, 1],
                                                     dtype=np.uint8))
