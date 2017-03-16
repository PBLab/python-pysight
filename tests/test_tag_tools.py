"""
__author__ = Hagai Hargil
"""

import unittest
import numpy as np
import pandas as pd


class TestTAGTools(unittest.TestCase):
    """
    Tests for TAG analysis functions
    """
    def test_tag_digitize(self):
        from pysight.tag_tools import numba_digitize

        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

        real_result = np.array([1, 4, 3, 2])
        result, _ = numba_digitize(x, bins)

        self.assertTrue(np.array_equal(real_result, result))

    def test_tag_verify(self):
        from pysight.tag_tools import verify_periodicity

        diff = 0.1897e6
        jitter = 0.05
        allowed_noise = np.ceil(jitter * 6550).astype(np.uint64)

        full_vec = np.arange(0, 200 * 6530, 6530)
        missing = [3, 17, 18, 49, 50, 51, 125, 129, 130, 165, 166, 169]
        full_vec_with_miss = full_vec[:-10].copy()
        full_vec_with_miss[missing] = -1
        missing_vec = pd.Series(full_vec_with_miss[full_vec_with_miss != -1], dtype=np.uint64)
        returned = verify_periodicity(tag_data=missing_vec, tag_freq=diff, binwidth=800e-12).astype(int)

        full_vec_as_series = pd.Series(full_vec, dtype=np.uint64)
        subtract = np.abs(full_vec_as_series.values[:-10] - returned.values[:-9])
        subtract[subtract < allowed_noise] = 0

        self.assertTrue(np.array_equal(subtract, np.zeros_like(full_vec[:-10])))

        pass

    if __name__ == '__main__':
        unittest.main()
