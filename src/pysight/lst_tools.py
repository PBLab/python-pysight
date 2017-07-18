import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple, List
from numba import jit, int64, uint64
import warnings
from pysight.apply_df_funcs import get_lost_bit_np, get_lost_bit_tag, iter_string_hex_to_bin, convert_hex_to_int, convert_hex_to_bin
from pysight.validation_tools import validate_line_input, validate_frame_input, \
    validate_laser_input, validate_created_data_channels, rectify_photons_in_uneven_lines, \
    calc_last_event_time
from attr.validators import instance_of
import sys
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt



@attr.s(slots=True)
class Analysis(object):
    """
    Create pipeline of analysis of lst files.
    """
    # TODO: Variable documentation
    dict_of_inputs     = attr.ib(validator=instance_of(dict))
    data               = attr.ib(validator=instance_of(np.ndarray))
    dict_of_slices_hex = attr.ib(validator=instance_of(dict))
    dict_of_slices_bin = attr.ib()
    num_of_channels    = attr.ib(default=1, validator=instance_of(int))
    timepatch          = attr.ib(default='32', validator=instance_of(str))
    data_range         = attr.ib(default=1, validator=instance_of(int))
    is_binary          = attr.ib(default=False, validator=instance_of(bool))
    num_of_frames      = attr.ib(default=1, validator=instance_of(int))
    x_pixels           = attr.ib(default=512, validator=instance_of(int))
    y_pixels           = attr.ib(default=512, validator=instance_of(int))
    laser_freq         = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth           = attr.ib(default=800e-12, validator=instance_of(float))
    bidir              = attr.ib(default=False, validator=instance_of(bool))
    tag_freq           = attr.ib(default=189e5, validator=instance_of(float))
    tag_pulses         = attr.ib(default=1, validator=instance_of(int))
    phase              = attr.ib(default=-2.6, validator=instance_of(float))
    use_tag_bits       = attr.ib(default=False, validator=instance_of(bool))
    laser_offset       = attr.ib(default=0.0, validator=instance_of(float))
    use_sweeps         = attr.ib(default=False, validator=instance_of(bool))
    keep_unidir        = attr.ib(default=False, validator=instance_of(bool))
    flim               = attr.ib(default=False, validator=instance_of(bool))
    exp_params         = attr.ib(default={}, validator=instance_of(dict))
    censor             = attr.ib(default=False, validator=instance_of(bool))
    time_after_sweep   = attr.ib(default=int(96), validator=instance_of(int))
    acq_delay          = attr.ib(default=int(0), validator=instance_of(int))
    line_freq          = attr.ib(default=7930.0, validator=instance_of(float))
    line_delta         = attr.ib(init=False)
    df_allocated       = attr.ib(init=False)
    dict_of_data       = attr.ib(init=False)
    data_to_grab       = attr.ib(init=False)


    def run(self):
        """ Pipeline of analysis """
        if self.is_binary:
            df_after_timepatch = self.tabulate_input_binary()
        else:
            df_after_timepatch = self.tabulate_input_hex()

        print('Sorted dataframe created. Starting setting the proper data channel distribution...')
        self.dict_of_data = self.determine_data_channels(df=df_after_timepatch)
        print('Channels of events found. Allocating photons to their frames and lines...')
        df_photons = self.__create_photon_dataframe()
        # Unidirectional scan - create fake lines
        if not self.bidir:
            self.add_unidirectional_lines()
        self.df_allocated = self.allocate_photons(df_photons=df_photons)
        print('Relative times calculated. Creating Movie object...')
        # Censor correction addition:
        if 'Laser' not in self.dict_of_data.keys():
            self.dict_of_data['Laser'] = 0
        # if self.use_sweeps:
        #     out = self.train_dataset()

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

    @property
    def offset(self):
        return int(np.floor(self.laser_offset * 10**-9 / self.binwidth))

    def __allocate_data_by_channel(self, df):
        """
        Go over the channels and find the events from that specific channel, assigning
        them to a dictionary with a suitable name.
        :param df: DataFrame with data to allocate.
        :return: Dict containing the data
        """

        dict_of_data = {}
        self.data_to_grab = ['abs_time']
        if self.use_sweeps: # Sweeps as lines - generate a "fake" line signal
            self.data_to_grab.extend(('edge', 'sweep', 'time_rel_sweep'))
            sweep_vec = np.arange(df['sweep'].max() + 1, dtype=np.uint64)
            if len(sweep_vec) < 2:
                warnings.warn("All data was registered to a single sweep. Line data will be completely simulated.")
            else:
                dict_of_data['Lines'] = pd.DataFrame(sweep_vec * self.data_range,
                                                     columns=['abs_time'], dtype=np.uint64)

        for key in self.dict_of_inputs:
            relevant_values = df.loc[df['channel'] == self.dict_of_inputs[key], self.data_to_grab]
            # NUMBA SORT NOT WORKING:
            # sorted_vals = numba_sorted(relevant_values.values)
            # dict_of_data[key] = pd.DataFrame(sorted_vals, columns=['abs_time'])
            if key in ['PMT1', 'PMT2']:
                dict_of_data[key] = relevant_values.reset_index(drop=True)
                dict_of_data[key]['Channel'] = 1 if 'PMT1' == key else 2  # Channel is the spectral channel
            else:
                dict_of_data[key] = relevant_values.sort_values(by=['abs_time']).reset_index(drop=True)

        # Apply offset for laser
        # try:
        #     dict_of_data['Laser']['abs_time'] = dict_of_data['laser']['abs_time'] + self.offset
        #     dict_of_data['Laser']['time_rel_sweep'] = dict_of_data['laser']['time_rel_sweep'] + self.offset
        # except KeyError:
        #     pass
        # CURRENTLY DISREGARDING OFFSET VALUES, SINCE THE ABOVE OPERATION MIGHT SHIFT LASER PULSES FROM THEIR RELATIVE
        # CORRECT SWEEP.

        return dict_of_data

    def determine_data_channels(self, df: pd.DataFrame=None) -> Dict:
        """ Create a dictionary that contains the data in its ordered form."""

        if df.empty:
            raise ValueError('Received dataframe was empty.')

        dict_of_data = self.__allocate_data_by_channel(df=df)

        if 'Frames' in dict_of_data:
            self.num_of_frames = dict_of_data['Frames'].shape[0] + 1  # account for first frame

        # Validations
        last_event_time = calc_last_event_time(dict_of_data=dict_of_data, lines_per_frame=self.y_pixels)
        dict_of_data, self.line_delta = validate_line_input(dict_of_data=dict_of_data, num_of_lines=self.y_pixels,
                                                       num_of_frames=self.num_of_frames, line_freq=self.line_freq,
                                                       last_event_time=last_event_time,
                                                       cols_in_data=self.data_to_grab)
        dict_of_data = validate_frame_input(dict_of_data=dict_of_data,
                                            num_of_lines=self.y_pixels, binwidth=self.binwidth,
                                            last_event_time=last_event_time,
                                            cols_in_data=self.data_to_grab)
        try:
            dict_of_data['Laser'] = validate_laser_input(dict_of_data['Laser'], laser_freq=self.laser_freq,
                                                         binwidth=self.binwidth, offset=self.offset)
        except KeyError:
            pass

        validate_created_data_channels(dict_of_data)
        return dict_of_data

    def __create_photon_dataframe(self):
        """
        If a single PMT channel exists, create a df_photons object.
        Else, concatenate the two data channels into a single dataframe.
        :return pd.DataFrame: Photon data
        """
        try:
            df_photons = pd.concat([self.dict_of_data['PMT1'].copy(),
                                    self.dict_of_data['PMT2'].copy()], axis=0)
            self.num_of_channels = 2
        except KeyError:
            df_photons = self.dict_of_data['PMT1'].copy()
        except:
            print("Unknown error: ", sys.exc_info()[0])
        finally:
            df_photons.loc[:, 'Channel'] = df_photons.loc[:, 'Channel'].astype('category')
            df_photons.set_index(keys='Channel', inplace=True)

        return df_photons

    def allocate_photons(self, df_photons: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe in which each photon is a part of a frame, line and possibly laser pulse
        :param dict_of_data: All events data, distributed to its input channel
        :param gui: Input GUI
        :return: pandas.DataFrame
        """
        from pysight.tag_tools import interpolate_tag

        # Preparations
        irrelevant_keys = {'PMT1', 'PMT2', 'TAG Lens'}
        relevant_keys = set(self.dict_of_data.keys()) - irrelevant_keys
        column_heads = {'Lines': 'time_rel_line_pre_drop', 'Frames': 'time_rel_frames', 'Laser': 'time_rel_pulse'}

        # Main loop - Sort lines and frames for all photons and calculate relative time
        for key in reversed(sorted(relevant_keys)):
            sorted_indices = numba_search_sorted(self.dict_of_data[key].loc[:, 'abs_time'].values,
                                                 df_photons.loc[:, 'abs_time'].values)
            try:
                df_photons[key] = self.dict_of_data[key].iloc[sorted_indices, 0].values  # columns 0 is abs_time,
                # but this .iloc method is amazingly faster than .loc
            except KeyError:
                warnings.warn(f'All computed sorted_indices were "-1" for key {key}. Trying to resume...')
            df_photons.dropna(how='any', inplace=True)
            df_photons.loc[:, key] = df_photons.loc[:, key].astype(np.uint64)
            # relative time of each photon in accordance to the line\frame\laser pulse
            df_photons[column_heads[key]] = df_photons['abs_time'] - df_photons[key]

            if 'Lines' == key:
                df_photons = rectify_photons_in_uneven_lines(df=df_photons,
                                                             sorted_indices=sorted_indices[sorted_indices >= 0],
                                                             lines=self.dict_of_data['Lines'].loc[:, 'abs_time'],
                                                             bidir=self.bidir, phase=self.phase,
                                                             keep_unidir=self.keep_unidir)

            if 'Laser' != key:
                df_photons.loc[:, key] = df_photons.loc[:, key].astype('category')
            df_photons.set_index(keys=key, inplace=True, append=True, drop=True)

        assert len(df_photons) > 0
        assert np.all(df_photons.iloc[:, 0].values >= 0)  # finds NaNs as well

        # Deal with TAG lens interpolation
        try:
            tag = self.dict_of_data['TAG Lens'].loc[:, 'abs_time']
        except KeyError:
            pass
        else:
            print('Interpolating TAG lens data...')
            df_photons = interpolate_tag(df_photons=df_photons, tag_data=tag, tag_freq=self.tag_freq,
                                         binwidth=self.binwidth, tag_pulses=self.tag_pulses)
            print('TAG lens interpolation finished.')

        # Deal with laser pulses interpolation
        if self.flim:
            df_photons, rel_time = self.__interpolate_laser(df_photons)
            if self.censor:
                self.exp_params = self.__fit_data_to_exponent(rel_time)

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

        if not self.use_tag_bits:
            self.dict_of_slices_hex.pop('tag', None)

        # Channel and edge information
        edge, channel = self.process_chan_edge(self.dict_of_slices_hex.pop('chan_edge'))
        # TODO: Timepatch == '3' is not supported because of this loop.

        if self.dict_of_slices_hex['lost'] is True:
            for key in list(self.dict_of_slices_hex.keys())[1:]:
                if self.dict_of_slices_hex[key].needs_bits:
                    list_with_lost = iter_string_hex_to_bin("".join(self.dict_of_slices_hex[key].data))
                    step_size = self.dict_of_slices_hex[key].end - self.dict_of_slices_hex[key].start
                    if key == 'tag':
                        list_of_losts, self.dict_of_slices_hex[key].processed = get_lost_bit_tag(list_with_lost,
                                                                                                 step_size,
                                                                                                 len(self.data))
                    else:
                        list_of_losts, self.dict_of_slices_hex[key].processed = get_lost_bit_np(list_with_lost,
                                                                                                step_size,
                                                                                                len(self.data))
                else:
                    if key == 'tag':
                        self.dict_of_slices_hex[key].processed = convert_hex_to_bin(self.dict_of_slices_hex[key].data)
                    else:
                        self.dict_of_slices_hex[key].processed = convert_hex_to_int(self.dict_of_slices_hex[key].data)
        else:
            for key in list(self.dict_of_slices_hex.keys())[1:]:
                if key == 'tag':
                    self.dict_of_slices_hex[key].processed = convert_hex_to_bin(self.dict_of_slices_hex[key].data)
                else:
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
        df['sweep'] = np.uint64(0)
        df['time_rel_sweep'] = np.uint64(0)

        if 'sweep' in self.dict_of_slices_hex:
            # Each event has an acquisition delay before the start of sweep time,
            # and has to be multiplied by the sweep number and the time-after-sweep delay
            df['abs_time'] = self.dict_of_slices_hex['time_rel_sweep'].processed + \
                ((self.dict_of_slices_hex['sweep'].processed - 1) * (self.data_range + self.time_after_sweep)) + \
                self.dict_of_slices_hex['sweep'].processed * self.acq_delay
            df['sweep'] = self.dict_of_slices_hex['sweep'].processed - 1
            df['time_rel_sweep'] = self.dict_of_slices_hex['time_rel_sweep'].processed
        else:
            df['abs_time'] = self.dict_of_slices_hex['time_rel_sweep'].processed

        # Before sorting all photons make sure that no input is missing from the user. If it's missing
        # the code will ignore this channel, but not raise an exception
        actual_data_channels = set(df['channel'].cat.categories.values)
        if actual_data_channels != set(self.dict_of_inputs.values()):
            warnings.warn("Channels that were inserted in GUI don't match actual data channels recorded. \n"
                          f"The list files contains data in the following channels: {actual_data_channels}.")

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
                popt, pcov = curve_fit(self.__nano_flim_exp, x, y_filt, p0=(max_value, 1/3.5, min_value),
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

        length_of_lines       = self.dict_of_data['Lines'].shape[0]
        new_line_arr          = np.zeros(length_of_lines * 2 - 1)
        new_line_arr[::2]     = self.dict_of_data['Lines'].loc[:, 'abs_time'].values
        new_line_arr[1::2]    = self.dict_of_data['Lines'].loc[:, 'abs_time']\
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
            rel_time251 = df.xs(key=chan, level='Channel', drop_level=False)['abs_time'].values % TEN_MEGAHERTZ_IN_BINS
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
