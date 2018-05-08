"""
__author__ = Hagai Hargil
"""

from unittest import TestCase
from pysight.nd_hist_generator.line_signal_validators.validation_tools import *
import pandas as pd
import numpy as np


class TestFrame(TestCase):
    """
    Tests for the validation functions
    """
    dict_of_data = dict(PMT1=pd.DataFrame([1, 10, 20, 30], columns=['abs_time']),
                        Lines=pd.DataFrame([0, 5, 10, 15, 20, 25, 30, 35], columns=['abs_time']))
    vlad = SignalValidator(dict_of_data)

    def test_last_event_only_pmt(self):
        dict_of_data = {'PMT1': pd.DataFrame([1, 5, 3], columns=['abs_time'])}
        last = SignalValidator(dict_of_data)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 5)

    def test_last_event_two_pmts(self):
        dict_of_data = {'PMT1': pd.DataFrame([1, 5, 3], columns=['abs_time']),
                        'PMT2': pd.DataFrame([1, 2, 3], columns=['abs_time'])}
        last = SignalValidator(dict_of_data)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 5)

    def test_last_event_with_frames(self):
        frame_data = pd.DataFrame([0, 100], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Frames': frame_data,
                        'Lines': [1, 2, 3]}
        last = SignalValidator(dict_of_data)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 200)

    def test_last_event_with_single_frame(self):
        frame_data = pd.DataFrame([100], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Frames': frame_data,
                        'Lines': [1, 2, 3]}
        last = SignalValidator(dict_of_data)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 200)

    def test_last_event_with_lines_less_than_needed_single_frame(self):
        line_data = pd.DataFrame([0, 10, 20], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        last = SignalValidator(dict_of_data, num_of_lines=5)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 50)

    def test_last_event_with_lines_less_than_needed_more_frames(self):
        line_data = pd.DataFrame([0, 10, 20, 30, 40], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        last = SignalValidator(dict_of_data, num_of_lines=3)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 30)

    def test_last_event_with_lines_just_like_needed(self):
        line_data = pd.DataFrame([0, 10, 20], columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        last = SignalValidator(dict_of_data, num_of_lines=3)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 30)

    def test_last_event_with_lines_more_than_needed(self):
        line_data = pd.DataFrame(np.arange(0, 100, 10), columns=['abs_time'])
        dict_of_data = {'PMT1': pd.DataFrame([1, 2, 3], columns=['abs_time']),
                        'Lines': line_data}
        last = SignalValidator(dict_of_data, num_of_lines=3)._SignalValidator__calc_last_event_time()
        self.assertEqual(last, 90)

    def test_bins_bet_lines(self):
        line_freq = 10.
        binwidth = 0.01
        bidir = False
        bins = SignalValidator(dict_of_data=self.dict_of_data, line_freq=line_freq, binwidth=binwidth,
                               bidir=bidir)._SignalValidator__bins_bet_lines()
        self.assertEqual(10, bins)

    def test_bins_bet_lines_bidir(self):
        line_freq = 10.
        binwidth = 0.01
        bidir = True
        bins = SignalValidator(dict_of_data=self.dict_of_data, line_freq=line_freq, binwidth=binwidth,
                               bidir=bidir)._SignalValidator__bins_bet_lines()
        self.assertEqual(5, bins)

    def test_extrapolate_without_zero(self):
        sigval = SignalValidator(self.dict_of_data, line_freq=0.5, binwidth=1.,
                                 delay_between_frames=0.)
        line_point = self.dict_of_data['Lines'].iat[1, 0]
        sigval.last_event_time = 15
        sigval.line_delta = sigval._SignalValidator__bins_bet_lines()
        sigval.num_of_lines = int(sigval.last_event_time // sigval.line_delta)
        returned_lines = sigval._SignalValidator__extrapolate_line_data(line_point=line_point)
        real_lines = np.arange(1, 15, step=2, dtype=np.uint64)
        self.assertSequenceEqual(returned_lines.abs_time.tolist(), real_lines.tolist())

    def test_extrapolate_from_zero(self):
        sigval = SignalValidator(self.dict_of_data, line_freq=1., binwidth=1.,
                                 delay_between_frames=0.)
        line_point = self.dict_of_data['Lines'].iat[2, 0]
        sigval.last_event_time = 15
        sigval.line_delta = sigval._SignalValidator__bins_bet_lines()
        sigval.num_of_lines = int(sigval.last_event_time // sigval.line_delta)
        returned_lines = sigval._SignalValidator__extrapolate_line_data(line_point=line_point)
        real_lines = np.arange(0, 15, step=1, dtype=np.uint64)
        self.assertSequenceEqual(returned_lines.abs_time.tolist(), real_lines.tolist())

    def test_extrapolate_without_timepoint(self):
        sigval = SignalValidator(self.dict_of_data, line_freq=1., binwidth=1.,
                                 delay_between_frames=0.)
        sigval.last_event_time = 15
        sigval.line_delta = sigval._SignalValidator__bins_bet_lines()
        sigval.num_of_lines = int(sigval.last_event_time // sigval.line_delta)
        returned_lines = sigval._SignalValidator__extrapolate_line_data()
        real_lines = np.arange(0, 15, step=1, dtype=np.uint64)
        self.assertSequenceEqual(returned_lines.abs_time.tolist(), real_lines.tolist())

    def test_extrapolate_without_timepoint_and_zero(self):
        sigval = SignalValidator(self.dict_of_data, line_freq=0.5, binwidth=1.,
                                 delay_between_frames=0.)
        sigval.last_event_time = 15
        sigval.line_delta = sigval._SignalValidator__bins_bet_lines()
        sigval.num_of_lines = int(sigval.last_event_time // sigval.line_delta) + 1
        returned_lines = sigval._SignalValidator__extrapolate_line_data()
        real_lines = np.arange(0, 15, step=2, dtype=np.uint64)
        self.assertSequenceEqual(returned_lines.abs_time.tolist(), real_lines.tolist())

    def test_pipeline_single_frame_si(self):
        lines = [0, 5, 10, 15, 20, 25, 30, 70, 75]
        dict_of_data = dict(PMT1=pd.DataFrame([1, 10, 20, 30, 100], columns=['abs_time'], dtype=np.uint64),
                            Lines=pd.DataFrame(lines, columns=['abs_time'],
                                               dtype=np.uint64))
        binwidth = 1.
        line_freq = 0.2
        delay_bet_frames = 0.
        num_of_lines = 7
        sigval = SignalValidator(dict_of_data=dict_of_data, binwidth=binwidth,
                                 line_freq=line_freq, delay_between_frames=delay_bet_frames,
                                 num_of_lines=num_of_lines, image_soft=ImagingSoftware.SCANIMAGE.value)
        sigval.run()
        self.assertSequenceEqual(sigval.dict_of_data['Lines'].abs_time.tolist(), lines[:-2])
        self.assertSequenceEqual(sigval.dict_of_data['Frames'].abs_time.tolist(), [0])
        self.assertEqual(sigval.line_delta, 5)


    def test_pipeline_few_frames_si(self):
        lines = [0, 5, 10, 45, 50, 55, 90, 95, 100, 135, 140]
        dict_of_data = dict(PMT1=pd.DataFrame([1, 10, 20, 30, 100], columns=['abs_time'], dtype=np.uint64),
                            Lines=pd.DataFrame(lines, columns=['abs_time'], dtype=np.uint64))
        binwidth = 1.
        line_freq = 0.2
        delay_bet_frames = 0.
        num_of_lines = 3
        sigval = SignalValidator(dict_of_data=dict_of_data, binwidth=binwidth,
                                 line_freq=line_freq, delay_between_frames=delay_bet_frames,
                                 num_of_lines=num_of_lines, image_soft=ImagingSoftware.SCANIMAGE.value)
        sigval.run()
        self.assertSequenceEqual(sigval.dict_of_data['Lines'].abs_time.tolist(), lines[:-2])
        self.assertSequenceEqual(sigval.dict_of_data['Frames'].abs_time.tolist(), [0, 45, 90])
        self.assertEqual(sigval.line_delta, 5)

    def test_pipeline_single_frame_mscan(self):
        lines = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        dict_of_data = dict(PMT1=pd.DataFrame([1, 10, 20, 30, 100], columns=['abs_time'], dtype=np.uint64),
                            Lines=pd.DataFrame(lines, columns=['abs_time'],
                                               dtype=np.uint64))
        binwidth = 1.
        line_freq = 0.2
        delay_bet_frames = 0.
        num_of_lines = 7
        sigval = SignalValidator(dict_of_data=dict_of_data, binwidth=binwidth,
                                 line_freq=line_freq, delay_between_frames=delay_bet_frames,
                                 num_of_lines=num_of_lines, image_soft=ImagingSoftware.MSCAN.value)
        sigval.run()
        self.assertSequenceEqual(sigval.dict_of_data['Lines'].abs_time.tolist(), lines[:-2])
        self.assertSequenceEqual(sigval.dict_of_data['Frames'].abs_time.tolist(), [0])
        self.assertEqual(sigval.line_delta, 5)

    def test_pipeline_few_frames_mscan(self):
        lines = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        dict_of_data = dict(PMT1=pd.DataFrame([1, 10, 20, 30, 100, 200], columns=['abs_time'], dtype=np.uint64),
                            Lines=pd.DataFrame(lines, columns=['abs_time'], dtype=np.uint64))
        binwidth = 1.
        line_freq = 0.2
        delay_bet_frames = 0.
        num_of_lines = 3
        sigval = SignalValidator(dict_of_data=dict_of_data, binwidth=binwidth,
                                 line_freq=line_freq, delay_between_frames=delay_bet_frames,
                                 num_of_lines=num_of_lines, image_soft=ImagingSoftware.MSCAN.value)
        sigval.run()
        self.assertSequenceEqual(sigval.dict_of_data['Lines'].abs_time.tolist(), lines[:-1])
        self.assertSequenceEqual(sigval.dict_of_data['Frames'].abs_time.tolist(), [0, 15, 30, 45])
        self.assertEqual(sigval.line_delta, 5)
