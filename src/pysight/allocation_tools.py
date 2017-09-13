"""
__author__ = Hagai Hargil
"""
import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple, List
from numba import jit, int64, uint64
import warnings
from pysight.validation_tools import  rectify_photons_in_uneven_lines
from attr.validators import instance_of
import sys
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt


@attr.s(slots=True)
class Allocate(object):
    """
    Create pipeline of analysis of lst files.
    """
    # TODO: Variable documentation
    dict_of_inputs     = attr.ib(validator=instance_of(dict))
    df_photons         = attr.ib(validator=instance_of(pd.DataFrame))
    num_of_channels    = attr.ib(default=1, validator=instance_of(int))
    laser_freq         = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth           = attr.ib(default=800e-12, validator=instance_of(float))
    bidir              = attr.ib(default=False, validator=instance_of(bool))
    tag_freq           = attr.ib(default=189e5, validator=instance_of(float))
    tag_pulses         = attr.ib(default=1, validator=instance_of(int))
    phase              = attr.ib(default=-2.6, validator=instance_of(float))
    keep_unidir        = attr.ib(default=False, validator=instance_of(bool))
    flim               = attr.ib(default=False, validator=instance_of(bool))
    exp_params         = attr.ib(default={}, validator=instance_of(dict))
    censor             = attr.ib(default=False, validator=instance_of(bool))
    dict_of_data       = attr.ib(default={}, validator=instance_of(dict))

    def run(self):
        """ Pipeline of analysis """
        print('Channels of events found. Allocating photons to their frames and lines...')
        # Unidirectional scan - create fake lines
        if not self.bidir:
            self.add_unidirectional_lines()
        self.allocate_photons()
        print('Relative times calculated. Creating Movie object...')
        # Censor correction addition:
        if 'Laser' not in self.dict_of_data.keys():
            self.dict_of_data['Laser'] = 0
            # if self.use_sweeps:
            #     out = self.train_dataset()

    def allocate_photons(self):
        """
        Returns a dataframe in which each photon is a part of a frame, line and possibly laser pulse
        """
        from pysight.tag_tools import interpolate_tag

        # Preparations
        irrelevant_keys = {'PMT1', 'PMT2', 'TAG Lens'}
        relevant_keys = set(self.dict_of_data.keys()) - irrelevant_keys
        column_heads = {'Lines': 'time_rel_line_pre_drop', 'Frames': 'time_rel_frames', 'Laser': 'time_rel_pulse'}

        # Main loop - Sort lines and frames for all photons and calculate relative time
        for key in reversed(sorted(relevant_keys)):
            sorted_indices = numba_search_sorted(self.dict_of_data[key].loc[:, 'abs_time'].values,
                                                 self.df_photons.loc[:, 'abs_time'].values)
            self.df_photons[key] = self.dict_of_data[key].iloc[sorted_indices, 0].values  # columns 0 is abs_time,
            # but this .iloc method is amazingly faster than .loc
            positive_mask = sorted_indices >= 0
            # drop photons that came before the first line
            self.df_photons = self.df_photons.iloc[positive_mask].copy()
            # relative time of each photon in accordance to the line\frame\laser pulse
            self.df_photons[column_heads[key]] = self.df_photons['abs_time'] - self.df_photons[key]

            if 'Lines' == key:
                self.df_photons = rectify_photons_in_uneven_lines(df=self.df_photons,
                                                                  sorted_indices=sorted_indices[sorted_indices >= 0],
                                                                  lines=self.dict_of_data['Lines'].loc[:, 'abs_time'],
                                                                  bidir=self.bidir, phase=self.phase,
                                                                  keep_unidir=self.keep_unidir)

            if 'Laser' != key:
                self.df_photons.loc[:, key] = self.df_photons.loc[:, key].astype('category')
            self.df_photons.set_index(keys=key, inplace=True, append=True, drop=True)

        assert len(self.df_photons) > 0
        assert np.all(self.df_photons.iloc[:, 0].values >= 0)  # finds NaNs as well

        # Deal with TAG lens interpolation
        try:
            tag = self.dict_of_data['TAG Lens'].loc[:, 'abs_time']
        except KeyError:
            pass
        else:
            print('Interpolating TAG lens data...')
            self.df_photons = interpolate_tag(df_photons=self.df_photons, tag_data=tag, tag_freq=self.tag_freq,
                                         binwidth=self.binwidth, tag_pulses=self.tag_pulses)
            print('TAG lens interpolation finished.')

        # Deal with laser pulses interpolation
        if self.flim:
            self.df_photons, rel_time = self.__interpolate_laser(self.df_photons)
            if self.censor:
                self.exp_params = self.__fit_data_to_exponent(rel_time)

    def __nano_flim_exp(self, x, a, b, c):
        """ Exponential function for FLIM and censor correction """
        return a * np.exp(-b * x) + c

    def __fit_data_to_exponent(self, list_of_channels: List) -> Dict:
        """
        Take the data after modulu BINS_BETWEEN_PULSES, in each channel,
         and fit an exponential decay to it, with some lifetime.
        :return: (A, b, C): Parameters of the fit A * exp( -b * x ) + C as a numpy array,
        inside a dictionary with the channel number as its key.
        """
        params = {}
        for chan, data_of_channel in enumerate(list_of_channels, 1):
            yn = np.histogram(data_of_channel, 16)[0]
            peakind = find_peaks_cwt(yn, np.arange(1, 10))
            if self.__requires_censoring(data=yn[peakind[0]:]):
                min_value = min(yn[yn > 0])
                max_value = yn[peakind[0]]
                y_filt = yn[peakind[0]:]
                x = np.arange(len(y_filt))
                popt, pcov = curve_fit(self.__nano_flim_exp, x, y_filt, p0=(max_value, 1 / 3.5, min_value),
                                       maxfev=10000)
                params[chan] = popt
            else:
                continue

        return params

    def __requires_censoring(self, data: np.ndarray):
        """
        Method to determine if we should undergo the censor correction process
        :param data: Bins of histogram from their peak onward.
        :return: boolean value
        """
        diffs = np.diff(data)
        if len(diffs) == 0:
            return True
        if np.all(diffs <= 0):  # No censoring occurred
            return False
        else:
            first_idx = np.argwhere(diffs >= 0)[0][0]
            diffs2 = diffs[first_idx:]
            if np.all(diffs2 >= 0) or np.all(diffs2 <= 0):
                return True
            else:  # either false alarm, or a third photon is on its way
                self.__requires_censoring(diffs2)
                return True

    def add_unidirectional_lines(self):
        """
        For unidirectional scans fake line signals have to be inserted.
        :param dict_of_data: All data
        :return:
        """

        length_of_lines = self.dict_of_data['Lines'].shape[0]
        new_line_arr = np.zeros(length_of_lines * 2 - 1)
        new_line_arr[::2] = self.dict_of_data['Lines'].loc[:, 'abs_time'].values
        new_line_arr[1::2] = self.dict_of_data['Lines'].loc[:, 'abs_time'] \
                                 .rolling(window=2).mean()[1:]

        self.dict_of_data['Lines'] = pd.DataFrame(new_line_arr, columns=['abs_time'],
                                                  dtype='uint64')

    def __interpolate_laser(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Assign a time relative to a laser pulse for each photon.
        :param df: Dataframe with data for each photon.
        Assuming that the clock is synced to a 10 MHz signal.
        :return: Modified dataframe.
        """
        TEN_MEGAHERTZ_IN_BINS = 251
        rel_time = []
        for chan in range(1, self.num_of_channels + 1):
            rel_time251 = df.xs(key=chan, level='Channel', drop_level=False)[
                              'abs_time'].values % TEN_MEGAHERTZ_IN_BINS
            rel_time_per_pulse = rel_time251 % np.ceil(1 / (self.binwidth * self.laser_freq))
            rel_time.append(np.uint8(rel_time_per_pulse))
            df.loc[chan, 'time_rel_pulse'] = np.uint8(rel_time_per_pulse)

        return df, rel_time

@jit(nopython=True, cache=True)
def numba_sorted(arr: np.array) -> np.array:
    """
    Sort an array with Numba. CURRENTLY NOT WORKING
    """

    arr.sort()
    return arr.astype(np.uint64)

@jit((int64[:](uint64[:], uint64[:])), nopython=True, cache=True)
def numba_search_sorted(input_sorted, input_values):
    """ Numba-powered searchsorted function. """
    return np.searchsorted(input_sorted, input_values) - 1
