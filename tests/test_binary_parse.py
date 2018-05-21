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

    def test_time_with_tp0(self):
        timepatch = '0'
        data = np.array([0b10000, 0b110000,
                         0b11000000000001111,
                         0b101010011])
        binda = BinaryDataParser(data, timepatch)
        times = np.array([1, 3, 2048, 21], dtype=np.uint64)
        calced = binda._BinaryDataParser__get_time()
        np.testing.assert_array_equal(calced, times)

    def test_time_with_tp5(self):
        timepatch = '5'
        data = np.array([0b10000, 0b110000,
                         0b1000000000001111,
                         0b101010011,
                         0b1111010010110000110100010])
        binda = BinaryDataParser(data, timepatch)
        times = np.array([1, 3, 2048, 21, 955930], dtype=np.uint64)
        calced = binda._BinaryDataParser__get_time()
        np.testing.assert_array_equal(calced, times)

    def test_sweep_standard(self):
        timepatch = '5'
        data = np.array([0b00000000001_01010101010101010101_0101,
                         0b00000000111_01010101010101010101_0101,
                         0b10010101_01010101010101010101_0101,
                         0b11010101_01010101010101010101_0101,
                         0b00010101_01010101010101010101_0101])
        binda = BinaryDataParser(data, timepatch)
        sweeps = np.array([1, 7, 149, 213, 21], dtype=np.uint16)
        calced = binda._BinaryDataParser__get_sweep()
        np.testing.assert_array_equal(calced, sweeps)

    def test_tag_standard(self):
        timepatch = '5b'
        data = np.array([0b001_0001010101000001_0101101001010101010101010101_0101,
                         0b100_0001010101000001_0101101001010101010101010101_0101,
                         0b010101_0001010101000001_0101101001010101010101010101_0101],
                        dtype=np.uint64)
        binda = BinaryDataParser(data, timepatch)
        tag = np.array([1, 4, 21], dtype=np.uint16)
        calced = binda._BinaryDataParser__get_tag()
        np.testing.assert_array_equal(calced, tag)

    def test_lost_standard(self):
        timepatch = '5b'
        data = np.array([0b1_011010011001001_0001010101000001_0101101001010101010101010101_0101,
                         0b0_011010011001001_0001010101000001_0101101001010101010101010101_0101,
                         0b1_011010011001001_0001010101000001_0101101001010101010101010101_0101,
                         0b1],
                        dtype=np.uint64)
        binda = BinaryDataParser(data, timepatch)
        lost = np.array([1, 0, 1, 0], dtype=np.uint8)
        calced = binda._BinaryDataParser__get_lost()
        np.testing.assert_array_equal(calced, lost)

