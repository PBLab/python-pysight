import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple
from numba import jit, int64, uint64
import warnings
from pysight.apply_df_funcs import get_lost_bit_np, iter_string_hex_to_bin, convert_hex_to_int
from pysight.validation_tools import validate_line_input, validate_frame_input, \
    validate_laser_input, validate_created_data_channels, rectify_photons_in_uneven_lines
from collections import defaultdict


@attr.s(slots=True)
class Analysis(object):
    """
    Create pipeline of analysis of lst files.
    """
    # TODO: Input validations
    # TODO: Variable documentation
    timepatch = attr.ib(validator=attr.validators.instance_of(str))
    data_range = attr.ib(validator=attr.validators.instance_of(int))
    dict_of_inputs = attr.ib()
    data = attr.ib()
    is_binary = attr.ib(validator=attr.validators.instance_of(bool))
    num_of_frames = attr.ib(validator=attr.validators.instance_of(int))
    x_pixels = attr.ib(validator=attr.validators.instance_of(int))
    y_pixels = attr.ib(validator=attr.validators.instance_of(int))
    laser_freq = attr.ib(validator=attr.validators.instance_of(float))
    binwidth = attr.ib(validator=attr.validators.instance_of(float))
    dict_of_slices_hex = attr.ib()
    dict_of_slices_bin = attr.ib()
    bidir = attr.ib(validator=attr.validators.instance_of(int))
    tag_freq = attr.ib(validator=attr.validators.instance_of(float))
    tag_pulses = attr.ib(validator=attr.validators.instance_of(int))
    phase = attr.ib(validator=attr.validators.instance_of(float))
    keep_unidir = attr.ib(default=False)
    df_allocated = attr.ib(init=False)
    dict_of_data = attr.ib(init=False)

    def run(self):
        """ Pipeline of analysis """
        if self.is_binary:
            df_after_timepatch = self.tabulate_input_binary()
        else:
            df_after_timepatch = self.tabulate_input_hex()

        print('Sorted dataframe created. Starting setting the proper data channel distribution...')
        self.dict_of_data, line_delta = self.determine_data_channels(df=df_after_timepatch)
        print('Channels of events found. Allocating photons to their frames and lines...')
        self.df_allocated = self.allocate_photons(dict_of_data=self.dict_of_data, line_delta=line_delta)
        print('Relative times calculated. Creating Movie object...')
        # Censor correction addition:
        if 'Laser' not in self.dict_of_data.keys():
            self.dict_of_data['Laser'] = 0

    @staticmethod
    def hex_to_bin_dict():
        """
        Create a simple dictionary that maps a hex input into a 4 letter binary output.
        :return: dict
        """
        diction = \
            {
                '0': '0000',
                '1': '0001',
                '2': '0010',
                '3': '0011',
                '4': '0100',
                '5': '0101',
                '6': '0110',
                '7': '0111',
                '8': '1000',
                '9': '1001',
                'a': '1010',
                'b': '1011',
                'c': '1100',
                'd': '1101',
                'e': '1110',
                'f': '1111',
            }
        return diction

    def determine_data_channels(self, df: pd.DataFrame=None) -> Tuple:
        """ Create a dictionary that contains the data in its ordered form."""

        if df.empty:
            raise ValueError('Received dataframe was empty.')

        dict_of_data = {}
        for key in self.dict_of_inputs:
            relevant_values = df.loc[df['channel'] == self.dict_of_inputs[key], 'abs_time']
            # NUMBA SORT NOT WORKING:
            # sorted_vals = numba_sorted(relevant_values.values)
            # dict_of_data[key] = pd.DataFrame(sorted_vals, columns=['abs_time'])
            if 'PMT1' == key or 'PMT2' == key:
                dict_of_data[key] = relevant_values.reset_index(drop=True)
            else:
                dict_of_data[key] = relevant_values.sort_values().reset_index(drop=True)

        # Validations
        dict_of_data, line_delta = validate_line_input(dict_of_data=dict_of_data, num_of_lines=self.y_pixels,
                                                       num_of_frames=self.num_of_frames)
        dict_of_data = validate_frame_input(dict_of_data=dict_of_data, line_delta=line_delta,
                                            num_of_lines=self.y_pixels, binwidth=self.binwidth)
        try:
            dict_of_data['Laser'] = validate_laser_input(dict_of_data['Laser'], laser_freq=self.laser_freq,
                                                         binwidth=self.binwidth)
        except KeyError:
            pass

        validate_created_data_channels(dict_of_data)
        return dict_of_data, line_delta

    def allocate_photons(self, dict_of_data=None, line_delta: float = -1) -> pd.DataFrame:
        """
        Returns a dataframe in which each photon is a part of a frame, line and possibly laser pulse
        :param dict_of_data: All events data, distributed to its input channel
        :param gui: Input GUI
        :return: pandas.DataFrame
        """
        from pysight.tag_tools import interpolate_tag

        # Preparations
        irrelevant_keys = {'PMT1', 'PMT2', 'TAG Lens'}
        relevant_keys = set(dict_of_data.keys()) - irrelevant_keys

        df_photons = dict_of_data['PMT1']  # TODO: Support more than one data channel
        df_photons = pd.DataFrame(df_photons, dtype='uint64')  # before this change it's a series with a name, not column head
        column_heads = {'Lines': 'time_rel_line_pre_drop', 'Frames': 'time_rel_frames', 'Laser': 'time_rel_pulse'}

        # Unidirectional scan - create fake lines
        if not self.bidir:
            dict_of_data = self.add_unidirectional_lines(dict_of_data=dict_of_data, line_delta=line_delta)

        # Main loop - Sort lines and frames for all photons and calculate relative time
        for key in reversed(sorted(relevant_keys)):
            sorted_indices = numba_search_sorted(dict_of_data[key].values, df_photons['abs_time'].values)
            try:
                df_photons[key] = dict_of_data[key].loc[sorted_indices].values
            except KeyError:
                warnings.warn('All computed sorted_indices were "-1" for key {}. Trying to resume...'.format(key))

            df_photons.dropna(how='any', inplace=True)
            df_photons[key] = df_photons[key].astype(np.uint64)
            df_photons[column_heads[key]] = df_photons['abs_time'] - df_photons[key] # relative time of each photon in
            #                                                                          accordance to the line\frame\laser pulse
            if 'Lines' == key:
                df_photons = rectify_photons_in_uneven_lines(df=df_photons,
                                                             sorted_indices=sorted_indices[sorted_indices >= 0],
                                                             lines=dict_of_data['Lines'], bidir=self.bidir,
                                                             phase=self.phase, keep_unidir=self.keep_unidir)

            if 'Laser' != key:
                df_photons[key] = df_photons[key].astype('category')
            df_photons.set_index(keys=key, inplace=True, append=True, drop=True)

        # Closure
        assert np.all(df_photons.values >= 0)  # finds NaNs as well
        # Deal with TAG lens interpolation
        try:
            tag = dict_of_data['TAG Lens']
        except KeyError:
            pass
        else:
            print('Interpolating TAG lens data...')
            df_photons = interpolate_tag(df_photons=df_photons, tag_data=tag, tag_freq=self.tag_freq,
                                         binwidth=self.binwidth, tag_pulses=self.tag_pulses)
            print('TAG lens interpolation finished.')

        # df_photons.drop(['abs_time'], axis=1, inplace=True)

        return df_photons

    def process_chan_edge(self, struct_of_data):
        """
        Simple processing scheme for the channel and edge data.
        """
        bin_array = np.array(iter_string_hex_to_bin("".join(struct_of_data.data)))
        edge      = self.slice_string_arrays(bin_array, start=0, end=1)
        channel   = self.slice_string_arrays(bin_array, start=1, end=4)

        return edge, channel

    def tabulate_input_hex(self) -> pd.DataFrame:
        """
        Reformat the read hex data into a dataframe.
        """

        for key in list(self.dict_of_slices_hex.keys())[1:]:
            self.dict_of_slices_hex[key].data = self.slice_string_arrays(self.data, self.dict_of_slices_hex[key].start,
                                                                         self.dict_of_slices_hex[key].end)

        # Channel and edge information
        edge, channel = self.process_chan_edge(self.dict_of_slices_hex.pop('chan_edge'))
        # TODO: Timepatch == '3' is not supported because of this loop.

        if self.dict_of_slices_hex['lost'] is True:
            for key in list(self.dict_of_slices_hex.keys())[1:]:
                if self.dict_of_slices_hex[key].needs_bits:
                    list_with_lost = iter_string_hex_to_bin("".join(self.dict_of_slices_hex[key].data))
                    step_size = self.dict_of_slices_hex[key].end - self.dict_of_slices_hex[key].start
                    list_of_losts, self.dict_of_slices_hex[key].processed = get_lost_bit_np(list_with_lost, step_size,
                                                                                            len(self.data))
                else:
                    self.dict_of_slices_hex[key].processed = convert_hex_to_int(self.dict_of_slices_hex[key].data)
        else:
            for key in list(self.dict_of_slices_hex.keys())[1:]:
                self.dict_of_slices_hex[key].processed = convert_hex_to_int(self.dict_of_slices_hex[key].data)

        # Reformat data
        df = pd.DataFrame(channel, columns=['channel'], dtype='category')
        df['edge'] = edge
        df['edge'] = df['edge'].astype('category')

        try:
            df['tag'] = self.dict_of_slices_hex['tag'].processed
        except KeyError:
            pass

        try:
            df['lost'] = list_of_losts  # TODO: Currently the LOST bit is meaningless
            df['lost'] = df['lost'].astype('category')
        except NameError:
            pass

        df['abs_time'] = np.uint64(0)

        if 'sweep' in self.dict_of_slices_hex:
            df['abs_time'] = self.dict_of_slices_hex['abs_time'].processed + (
                (self.dict_of_slices_hex['sweep'].processed - 1) * self.data_range)
        else:
            df['abs_time'] = self.dict_of_slices_hex['abs_time'].processed

        # Before sorting all photons make sure that no input is missing from the user. If it's missing
        # the code will ignore this channel, but not raise an exception
        actual_data_channels = set(df['channel'].cat.categories.values)
        if actual_data_channels != set(self.dict_of_inputs.values()):
            warnings.warn("Channels that were inserted in GUI don't match actual data channels recorded. \n"
                          "The list files contains data in the following channels: {}.".format(actual_data_channels))

        assert np.all(df['abs_time'].values >= 0)

        return df

    @staticmethod
    def slice_string_arrays(arr: np.array, start: int, end: int) -> np.array:
        """
        Slice an array of strings efficiently.
        Based on http://stackoverflow.com/questions/39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
        with modifications for Python 3.
        """
        b = arr.view('U1').reshape(len(arr), -1)[:, start:end]
        return np.fromstring(b.tostring(), dtype='U' + str(end - start))

    def tabulate_input_binary(self) -> pd.DataFrame:
        """
        Reformat the read binary data into a dataframe.
        """
        num_of_lines = self.data.shape[0]

        for key in self.dict_of_slices_bin:
            cur_data = self.data[:, self.dict_of_slices_bin[key].start:self.dict_of_slices_bin[key].end]
            try:
                zero_arr = np.zeros(num_of_lines, self.dict_of_slices_bin[key].cols)

            except AttributeError:  # No cols field since number of bits is a multiple of 8
                # self.dict_of_slices_bin[key].data_as_
                pass

    @staticmethod
    def add_unidirectional_lines(dict_of_data: Dict, line_delta: float = -1):
        """
        For unidirectional scans fake line signals have to be inserted.
        :param dict_of_data: All data
        :param line_delta: difference between lines
        :return:
        """

        if line_delta == -1:
            raise ValueError('Line delta variable was miscalculated.')

        length_of_lines       = dict_of_data['Lines'].shape[0]
        new_line_arr          = np.zeros(length_of_lines * 2 - 1)
        new_line_arr[::2]     = dict_of_data['Lines'].values
        new_line_arr[1::2]    = dict_of_data['Lines'].rolling(window=2).mean()[1:]

        dict_of_data['Lines'] = pd.Series(new_line_arr, name='Lines', dtype='uint64')
        return dict_of_data


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
