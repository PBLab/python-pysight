"""
__author__ = Hagai Har-Gil
"""
from unittest import TestCase
import pathlib

import zarr
import toml
import numpy as np

from pysight.main import main_data_readout


class TestEntirePipeline(TestCase):
    """ Integration tests for the entire application """

    def test_standard_pipeline(self):
        cfg_file = str(next(pathlib.Path(".").rglob("*jul.toml")).absolute())
        data_file = str(next(pathlib.Path(".").rglob("*jul.zarr")).absolute())
        with open(cfg_file, "r") as f:
            config: Dict[str, Any] = toml.load(f)
        out = main_data_readout(config)
        z = zarr.open(data_file, "r")
        np.testing.assert_array_equal(
            out.ch1.time_summed, np.array(z["/Summed Stack/Channel 1"])
        )

    def test_mscan_pipeline(self):
        cfg_file = str(next(pathlib.Path(".").rglob("*fly.toml")).absolute())
        data_file = str(next(pathlib.Path(".").rglob("*fly.zarr")).absolute())
        with open(cfg_file, "r") as f:
            config: Dict[str, Any] = toml.load(f)
        out = main_data_readout(config)
        z = zarr.open(data_file, "r")
        np.testing.assert_array_equal(
            out.ch1.time_summed, np.array(z["/Summed Stack/Channel 1"])
        )

    def test_tag_pipeline(self):
        cfg_file = str(next(pathlib.Path(".").rglob("*tag.toml")).absolute())
        data_file = str(next(pathlib.Path(".").rglob("*tag.zarr")).absolute())
        with open(cfg_file, "r") as f:
            config: Dict[str, Any] = toml.load(f)
        out = main_data_readout(config)
        z = zarr.open(data_file, "r")
        np.testing.assert_array_equal(
            out.ch1.time_summed, np.array(z["/Summed Stack/Channel 1"])
        )
