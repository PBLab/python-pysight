"""
__author__ = Hagai Hargil
"""
import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple, List
from attr.validators import instance_of
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt

from pysight.nd_hist_generator.tag_lens import TagPipeline


@attr.s(slots=True)
class Allocate(object):
    """
    Create pipeline of analysis of lst files.
    """
    # TODO: Variable documentation
    df_photons         = attr.ib(validator=instance_of(pd.DataFrame))
    dict_of_data       = attr.ib(validator=instance_of(dict))
    laser_freq         = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth           = attr.ib(default=800e-12, validator=instance_of(float))
    bidir              = attr.ib(default=False, validator=instance_of(bool))
    tag_freq           = attr.ib(default=189e3, validator=instance_of(float))
    tag_pulses         = attr.ib(default=1, validator=instance_of(int))
    phase              = attr.ib(default=-2.78, validator=instance_of(float))
    keep_unidir        = attr.ib(default=False, validator=instance_of(bool))
    flim               = attr.ib(default=False, validator=instance_of(bool))
    exp_params         = attr.ib(default={}, validator=instance_of(dict))
    censor             = attr.ib(default=False, validator=instance_of(bool))
    tag_interp_ok      = attr.ib(default=False, validator=instance_of(bool))
    tag_to_phase       = attr.ib(default=True, validator=instance_of(bool))
    tag_offset         = attr.ib(default=0, validator=instance_of(int))
    sorted_indices     = attr.ib(init=False)

    def run(self):
        """ Pipeline of analysis """
        print('Channels of events found. Allocating photons to their frames and lines...')
        # Unidirectional scan - create fake lines
        if not self.bidir:
            self.__add_unidirectional_lines()
        self.__allocate_photons()
        self.__allocate_tag()
        if self.flim:
            self.df_photons, rel_time = self.__interpolate_laser(self.df_photons)
            if self.censor:
                self.exp_params = self.__fit_data_to_exponent(rel_time)
        # Censor correction addition:
        if 'Laser' not in self.dict_of_data.keys():
            self.dict_of_data['Laser'] = 0
            # if self.use_sweeps:
            #     out = self.train_dataset()
        self.__reindex_dict_of_data()

        print('Relative times calculated. Creating Movie object...')

    @property
    def num_of_channels(self):
        return sum([1 for key in self.dict_of_data.keys() if 'PMT' in key])

    def __allocate_photons(self):
        """
        Returns a dataframe in which each photon is a part of a frame, line and possibly laser pulse
        """
        irrelevant_keys = {'PMT1', 'PMT2', 'TAG Lens'}
        relevant_keys = set(self.dict_of_data.keys()) - irrelevant_keys
        # Preparations
        column_heads = {'Lines': 'time_rel_line_pre_drop', 'Laser': 'time_rel_pulse'}

        # Main loop - Sort lines and frames for all photons and calculate relative time
        for key in reversed(sorted(relevant_keys)):
            sorted_indices = np.digitize(self.df_photons.loc[:, 'abs_time'].values,
                                         self.dict_of_data[key].loc[:, 'abs_time'].values) - 1
            self.df_photons[key] = self.dict_of_data[key].iloc[sorted_indices, 0].values  # columns 0 is abs_time,
            # but this .iloc method is amazingly faster than .loc
            positive_mask = sorted_indices >= 0
            # drop photons that came before the first line
            self.df_photons = self.df_photons.iloc[positive_mask].copy()
            # relative time of each photon in accordance to the line\frame\laser pulse
            if 'Frames' != key:
                self.df_photons[column_heads[key]] = self.df_photons['abs_time'] - self.df_photons[key]
            self.sorted_indices = sorted_indices[sorted_indices >= 0]
            if 'Lines' == key:
                self.__rectify_photons_in_uneven_lines()

            if 'Laser' != key:
                self.df_photons.loc[:, key] = self.df_photons.loc[:, key].astype('category')
            self.df_photons.set_index(keys=key, inplace=True, append=True, drop=True)

        assert len(self.df_photons) > 0
        assert np.all(self.df_photons.iloc[:, 0].values >= 0)  # finds NaNs as well

    def __allocate_tag(self):
        """ Allocate photons to TAG lens phase """
        try:
            tag = self.dict_of_data['TAG Lens'].loc[:, 'abs_time']
        except KeyError:
            return
        else:
            print('Interpolating TAG lens data...')
            tag_pipe = TagPipeline(photons=self.df_photons, tag_pulses=tag, freq=self.tag_freq,
                                   binwidth=self.binwidth, num_of_pulses=self.tag_pulses,
                                   to_phase=self.tag_to_phase, offset=self.tag_offset)
            tag_pipe.run()
            self.df_photons = tag_pipe.photons
            self.tag_interp_ok = tag_pipe.finished_pipe

            print('TAG lens interpolation finished.')

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

    def __add_unidirectional_lines(self):
        """
        For unidirectional scans fake line signals have to be inserted for us to identify forward- and
        back-phase photons.
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

    def __add_phase_offset_to_bidir_lines(self):
        """
        Uneven lines in a bidirectional scanning mode have to be offsetted.
        """
        phase_in_seconds = self.phase * 1e-6
        self.dict_of_data['Lines'].abs_time.iloc[1::2] -= np.uint64(phase_in_seconds/self.binwidth)

    def __rectify_photons_in_uneven_lines(self):
        """
        "Deal" with photons in uneven lines. Unidir - if keep_unidir is false, will throw them away.
        Bidir = flips them over (in the Volume object)
        """
        uneven_lines = np.remainder(self.sorted_indices, 2)
        if self.bidir:
            self.df_photons.rename(columns={'time_rel_line_pre_drop': 'time_rel_line'}, inplace=True)

        elif not self.bidir and not self.keep_unidir:
            self.df_photons = self.df_photons.iloc[uneven_lines != 1, :].copy()
            self.df_photons.rename(columns={'time_rel_line_pre_drop': 'time_rel_line'}, inplace=True)
            self.dict_of_data['Lines'] = self.dict_of_data['Lines'].iloc[::2, :].copy().reset_index()

        elif not self.bidir and self.keep_unidir:  # Unify the excess rows and photons in them into the previous row
            self.sorted_indices[np.logical_and(uneven_lines, 1)] -= 1
            self.df_photons.loc['Lines'] = self.dict_of_data['Lines'].loc[self.sorted_indices, 'abs_time'].values
            self.dict_of_data['Lines'] = self.dict_of_data['Lines'].iloc[::2, :].copy().reset_index()
        try:
            self.df_photons.drop(['time_rel_line_pre_drop'], axis=1, inplace=True)
        except (ValueError, KeyError):  # column label doesn't exist
            pass
        self.df_photons = self.df_photons.loc[self.df_photons.loc[:, 'time_rel_line'] >= 0, :]

    def  __reindex_dict_of_data(self):
        """
        Add new frame indices to the Series composing ``self.dict_of_data`` for slicing later on.
        The "Frames" indices are its values, and the "Lines" indices are the corresponding frames.
        """
        # Frames
        self.dict_of_data['Frames'] = pd.Series(self.dict_of_data['Frames'].abs_time.values,
                                                index=self.dict_of_data['Frames'].abs_time.values)
        # Lines
        lines = self.dict_of_data['Lines'].abs_time.values
        sorted_indices = np.digitize(lines, self.dict_of_data['Frames'].values) - 1
        positive_mask = sorted_indices >= 0
        lines = lines[positive_mask].copy()
        sorted_indices = sorted_indices[positive_mask]
        self.dict_of_data['Lines'] = pd.Series(lines, index=self.dict_of_data['Frames'].iloc[sorted_indices].values)
