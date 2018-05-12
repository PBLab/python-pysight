"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
import numpy as np
import pathlib
import h5py
import tkinter

from pysight.main import run


class TestEntirePipeline(TestCase):
    """ Integration tests for the entire application """

    def test_standard_pipeline(self):
        cfg_file = str(next(pathlib.Path('.').rglob('*jul.json')).absolute())
        data_file = str(next(pathlib.Path('.').rglob('*jul.hdf5')).absolute())
        df, movie = run(cfg_file)
        with h5py.File(data_file, 'r') as f:
            np.testing.assert_array_equal(movie.summed_mem[1], np.array(f['/Summed Stack/Channel 1']))

    def test_mscan_pipeline(self):
        cfg_file = str(next(pathlib.Path('.').rglob('*fly.json')).absolute())
        data_file = str(next(pathlib.Path('.').rglob('*fly.hdf5')).absolute())
        df, movie = run(cfg_file)
        with h5py.File(data_file, 'r') as f:
            np.testing.assert_array_equal(movie.summed_mem[1], np.array(f['/Summed Stack/Channel 1']))

    def test_tag_pipeline(self):
        cfg_file = str(next(pathlib.Path('.').rglob('*tag.json')).absolute())
        data_file = str(next(pathlib.Path('.').rglob('*tag.hdf5')).absolute())
        df, movie = run(cfg_file)
        with h5py.File(data_file, 'r') as f:
            np.testing.assert_array_equal(movie.summed_mem[1], np.array(f['/Summed Stack/Channel 1']))
