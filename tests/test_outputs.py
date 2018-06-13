"""
__author__ = Hagai Hargil
"""

from unittest import TestCase
from pysight.nd_hist_generator.outputs import *
from os import sep


class TestOutput(TestCase):
    """ Test how well we parse the outputs """
    parser_objects = []
    parser_objects.append(OutputParser(output_dict={}, filename="tests_data" + sep + f"output_1.hdf5",
                                       flim=True))
    parser_objects.append(OutputParser(output_dict={'memory': True}, filename="tests_data" + sep + f"output_2.hdf5",
                                       reprate=160.3e6, flim=True, z_pixels=10, num_of_frames=2))
    parser_objects.append(OutputParser(output_dict={}, filename="tests_data" + sep + f"output_1.hdf5",
                                       num_of_frames=2))

    def test_bins_bet_pulses(self):
        bins_bet_pulses = [16, 8, 1]
        for obj, bins in zip(self.parser_objects, bins_bet_pulses):
            self.assertEqual(obj.bins_bet_pulses, bins)


    def test_determine_shape(self):
        shapes = [(1, 512, 512, 16), (2, 512, 512, 10, 8), (2, 512, 512)]
        for shape, obj in zip(shapes, self.parser_objects):
            self.assertEqual(shape, obj.determine_data_shape_full())

    def test_non_squeezed_shapes(self):
        shapes = [(1, 512, 512, 16, 1), (1, 1, 512, 100), (10, 512, 1, 16), (10, 512, 1)]
        squeezed_shapes = [(1, 512, 512, 16), (1, 512, 100), (10, 512, 16), (10, 512)]
        output_obj = []
        output_obj.append(OutputParser(output_dict={}, filename="tests_data" + sep + f"output_3.hdf5",
                                       z_pixels=16, num_of_frames=1))
        output_obj.append(OutputParser(output_dict={}, filename="tests_data" + sep + f"output_4.hdf5",
                                       z_pixels=100, num_of_frames=1, x_pixels=1))
        output_obj.append(OutputParser(output_dict={}, filename="tests_data" + sep + f"output_5.hdf5",
                                       z_pixels=16, num_of_frames=10, y_pixels=1))
        output_obj.append(OutputParser(output_dict={}, filename="tests_data" + sep + f"output_6.hdf5",
                                       num_of_frames=10, y_pixels=1))
        for shape, obj in zip(squeezed_shapes, output_obj):
            self.assertEqual(shape, obj.determine_data_shape_full())
