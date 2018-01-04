"""
__author__ = Hagai Hargil
"""
from unittest import TestCase
from pysight.nd_hist_generator.line_signal_validators.rectify_lines import LineRectifier
import numpy as np


class TestLineRectifier(TestCase):

    def test_normal_line_signal_from_zero_unidir(self):
        """
        Non corrupt signal to test
        :return:
        """
        raw_lines = np.arange(start=0, stop=100, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        rect = LineRectifier(lines=raw_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_normal_line_signal_from_zero_bidir(self):
        """
        Non corrupt signal to test
        :return:
        """
        raw_lines = np.arange(start=0, stop=100, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        rect = LineRectifier(lines=raw_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify())[:-1], list(raw_lines)[:-1])

    def test_normal_line_signal_no_zero_unidir(self):
        """
        Non corrupt signal to test
        :return:
        """
        raw_lines = np.arange(start=10, stop=100, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        rect = LineRectifier(lines=raw_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_normal_line_signal_no_zero_bidir(self):
        """
        Non corrupt signal to test
        :return:
        """
        raw_lines = np.arange(start=10, stop=100, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        rect = LineRectifier(lines=raw_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_normal_line_signal_with_noise_unidir(self):
        """
        Non corrupt signal to test
        :return:
        """
        step = 100
        raw_lines = np.arange(start=500, stop=15000, step=step, dtype=np.uint64)
        raw_lines += ((np.random.rand(raw_lines.shape[0]) - 0.5) * step/25).astype(np.uint64)
        num_lines = len(raw_lines[:-1])
        rect = LineRectifier(lines=raw_lines[:-1], x_pixels=num_lines,
                             bidir=False)

        for idx, (val1, val2) in enumerate(zip(rect.rectify()[:-1], raw_lines[:-1])):
            self.assertGreaterEqual(1, np.abs(float(val1)-float(val2)))

    def test_normal_line_signal_with_noise_bidir(self):
        """
        Non corrupt signal to test
        :return:
        """
        step = 100
        raw_lines = np.arange(start=500, stop=15000, step=step, dtype=np.uint64)
        raw_lines += ((np.random.rand(raw_lines.shape[0]) - 0.5) * step/25).astype(np.uint64)
        num_lines = len(raw_lines[:-1])
        rect = LineRectifier(lines=raw_lines[:-1], x_pixels=num_lines,
                             bidir=True)

        for idx, (val1, val2) in enumerate(zip(rect.rectify()[:-1], raw_lines[:-1])):
            self.assertGreaterEqual(1, np.abs(float(val1)-float(val2)))

    def test_single_missing_line_middle_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, 2)
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_single_missing_line_middle_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, 2)
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_multiple_line_middle_single_instance_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [2, 3, 4])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_multiple_line_middle_single_instance_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [2, 3, 4])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_multiple_single_lines_middle_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [2, 10, 100, 200, 300, 234])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_multiple_single_lines_middle_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [2, 10, 100, 200, 300, 234])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_multiple_lines_middle_unidir(self):
        raw_lines = np.arange(start=0, stop=10000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [2, 10, 100, 101, 102, 200, 201, 300, 234])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_multiple_lines_middle_bidir(self):
        raw_lines = np.arange(start=0, stop=10000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [2, 10, 100, 101, 102, 200, 201, 300, 234])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_line_from_end_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [len(raw_lines)-2])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_line_from_end_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [len(raw_lines)-2])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_lines_from_end_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [len(raw_lines)-2, len(raw_lines)-3, len(raw_lines)-4])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_lines_from_end_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [len(raw_lines)-2, len(raw_lines)-3, len(raw_lines)-4])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_lines_from_end_and_middle_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [len(raw_lines)-2, len(raw_lines)-3, len(raw_lines)-4,
                                          10, 20, 21, 22, 51])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_lines_from_end_and_middle_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [len(raw_lines)-2, len(raw_lines)-3, len(raw_lines)-4,
                                          10, 20, 21, 22, 51])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_issue_with_first_diff_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [1, 2])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_issue_with_first_diff_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [1, 2])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_issue_with_first_diff_and_more_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [1, 2, 10, 40, 41, 42, 100, 101])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_issue_with_first_diff_and_more_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [1, 2, 10, 40, 41, 42, 100, 101])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_starting_lines_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [0, 1, 2, 3])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_starting_lines_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [0, 1, 2, 3])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_head_tail_unidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [0, 1, 2, 3, len(raw_lines)-2, len(raw_lines)-3])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        expected_lines = np.array([0, 3, 6, 10, 13, 16], dtype=np.uint64)
        expected_lines = np.concatenate((expected_lines, raw_lines[4:-2]))
        self.assertSequenceEqual(list(rect.rectify()), list(expected_lines))

    def test_missing_head_tail_bidir(self):
        raw_lines = np.arange(start=0, stop=1000, step=5, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [0, 1, 2, 3, len(raw_lines)-2, len(raw_lines)-3])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        expected_lines = np.array([0, 3, 6, 10, 13, 16], dtype=np.uint64)
        expected_lines = np.concatenate((expected_lines, raw_lines[4:-2]))
        self.assertSequenceEqual(list(rect.rectify()), list(expected_lines))

    def test_missing_with_one_line_ok_unidir(self):
        raw_lines = np.arange(start=0, stop=2000, step=50, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 5])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_one_line_ok_bidir(self):
        raw_lines = np.arange(start=0, stop=2000, step=50, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 5])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_two_lines_ok_unidir(self):
        raw_lines = np.arange(start=0, stop=2000, step=50, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 6])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_two_lines_ok_bidir(self):
        raw_lines = np.arange(start=0, stop=2000, step=50, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 6])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_two_lines_and_one_ok_unidir(self):
        raw_lines = np.arange(start=0, stop=2500, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 6, 9, 11])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_two_lines_and_one_ok_bidir(self):
        raw_lines = np.arange(start=0, stop=2500, step=10, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 6, 9, 11])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_two_lines_or_one_ok_unidir(self):
        raw_lines = np.arange(start=0, stop=2500, step=50, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 6, 11])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=False)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))

    def test_missing_with_two_lines_or_one_ok_bidir(self):
        raw_lines = np.arange(start=0, stop=2500, step=50, dtype=np.uint64)
        num_lines = len(raw_lines[:-1])
        del_lines = np.delete(raw_lines, [3, 6, 11])
        rect = LineRectifier(lines=del_lines[:-1], x_pixels=num_lines,
                             bidir=True)
        self.assertSequenceEqual(list(rect.rectify()), list(raw_lines))
