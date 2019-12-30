"""
__author__ = Hagai Hargil
"""

from unittest import TestCase
from pysight.nd_hist_generator.outputs import *
from os import sep

import pytest


class TestOutput(TestCase):
    """ Test how well we parse the outputs """

    parser_objects = []
    parser_objects.append(
        OutputParser(output_dict={}, filename="tests_data" + sep + "output_1.hdf5",)
    )
    parser_objects.append(
        OutputParser(
            output_dict={"memory": True},
            filename="tests_data" + sep + "output_2.hdf5",
            reprate=160.3e6,
            z_pixels=10,
            num_of_frames=2,
        )
    )
    parser_objects.append(
        OutputParser(
            output_dict={},
            filename="tests_data" + sep + "output_1.hdf5",
            num_of_frames=2,
        )
    )

    def test_determine_shape(self):
        shapes = [(1, 512, 512), (2, 512, 512, 10), (2, 512, 512)]
        for shape, obj in zip(shapes, self.parser_objects):
            self.assertEqual(shape, obj.determine_data_shape_full())

    def test_non_squeezed_shapes(self):
        shapes = [
            (1, 512, 512, 16, 1),
            (1, 1, 512, 100),
            (10, 512, 16),
            (10, 512, 1),
            (1, 512, 512, 100),
        ]
        squeezed_shapes = [
            (1, 512, 512, 16),
            (1, 512, 100),
            (10, 512, 16),
            (10, 512),
            (1, 512, 512, 100),
        ]
        output_obj = []
        output_obj.append(
            OutputParser(
                output_dict={},
                filename="tests_data" + sep + f"output_3.hdf5",
                z_pixels=16,
                num_of_frames=1,
            )
        )
        output_obj.append(
            OutputParser(
                output_dict={},
                filename="tests_data" + sep + f"output_4.hdf5",
                z_pixels=100,
                num_of_frames=1,
                x_pixels=1,
            )
        )
        output_obj.append(
            OutputParser(
                output_dict={},
                filename="tests_data" + sep + f"output_5.hdf5",
                z_pixels=16,
                num_of_frames=10,
                y_pixels=1,
            )
        )
        output_obj.append(
            OutputParser(
                output_dict={},
                filename="tests_data" + sep + f"output_6.hdf5",
                num_of_frames=10,
                y_pixels=1,
            )
        )
        output_obj.append(
            OutputParser(
                output_dict={},
                filename="tests_data" + sep + f"output_7.hdf5",
                num_of_frames=1,
                z_pixels=100,
            )
        )
        for shape, obj in zip(squeezed_shapes, output_obj):
            self.assertEqual(shape, obj.determine_data_shape_full())


@pytest.fixture
def generate_output_obj():
    def _generate(shape, flim, di=None):
        if di is None:
            di = {1: np.array([1, 2, 3])}
        photons = pd.DataFrame({1: np.array([1, 2, 3])})
        summed_mem = di
        stack = di
        channels = pd.CategoricalIndex([1])
        return PySightOutput(photons, summed_mem, stack, channels, shape, False, dict())

    return _generate


class TestPySightOutput:
    def test_data_shape_txy(self, generate_output_obj):
        shape = (10, 10, 10)
        flim = False
        obj = generate_output_obj(shape, flim)
        shape = DataShape(10, 10, 10, None)
        assert shape == obj._parse_data_shape()

    def test_data_shape_txyz(self, generate_output_obj):
        shape = (10, 10, 10, 10)
        flim = False
        di = {1: np.random.random(shape)}
        obj = generate_output_obj(shape, flim, di)
        shape = DataShape(10, 10, 10, 10)
        assert shape == obj._parse_data_shape()

    def test_data_shape_txyztau(self, generate_output_obj):
        shape = (10, 10, 10, 10)
        flim = True
        di = {1: np.random.random(shape)}
        obj = generate_output_obj(shape, flim, di)
        shape = DataShape(10, 10, 10, 10)
        assert shape == obj._parse_data_shape()

    @pytest.mark.xfail
    def test_data_shape_txyz_notau(self, generate_output_obj):
        shape = (10, 10, 10, 10, 5)
        flim = False
        di = {1: np.random.random(shape)}
        obj = generate_output_obj(shape, flim, di)
        shape = DataShape(10, 10, 10, 10)
        assert shape == obj._parse_data_shape()

    def test_data_shape_txy_tau(self, generate_output_obj):
        shape = (10, 10, 10, 5)
        flim = True
        di = {1: np.random.random(shape)}
        obj = generate_output_obj(shape, flim, di)
        shape = DataShape(10, 10, 10, 5)
        assert shape == obj._parse_data_shape()
