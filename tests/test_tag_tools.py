"""
__author__ = Hagai Hargil
"""

import unittest
import numpy as np

class TAGTools(unittest.TestCase):
    """
    Tests for TAG analysis functions
    """
    def test_tag_digitize(self):
        from pysight.tag_tools import numba_digitize

        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

        real_result = np.array([1, 4, 3, 2])
        result, _ = numba_digitize(x, bins)

        self.assertEqual((real_result, result))

    def test_tag_verify(self):
        from pysight.tag_tools import verify_periodicity

        pass

    if __name__ == '__main__':
        unittest.main()
