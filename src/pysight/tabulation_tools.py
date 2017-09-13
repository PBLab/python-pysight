"""
__author__ = Hagai Hargil
"""
import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple, List
import warnings
from pysight.apply_df_funcs import get_lost_bit_np, get_lost_bit_tag, iter_string_hex_to_bin, convert_hex_to_int, convert_hex_to_bin
from pysight.validation_tools import validate_line_input, validate_frame_input, \
    validate_laser_input, validate_created_data_channels, rectify_photons_in_uneven_lines, \
    calc_last_event_time
from attr.validators import instance_of


@attr.s(slots=True)
class Tabulate(object):
    """
    Prepare lst files so that they could be imaged using later modules.
    """
    # TODO: Variable documentation
    dict_of_inputs     = attr.ib(validator=instance_of(dict))
    data               = attr.ib(validator=instance_of(np.ndarray))
    dict_of_slices_hex = attr.ib(validator=instance_of(dict))
    dict_of_slices_bin = attr.ib()
    x_pixels           = attr.ib(default=512, validator=instance_of(int))
    y_pixels           = attr.ib(default=512, validator=instance_of(int))
    timepatch          = attr.ib(default='32', validator=instance_of(str))
    data_range         = attr.ib(default=1, validator=instance_of(int))
    is_binary          = attr.ib(default=False, validator=instance_of(bool))
    num_of_frames      = attr.ib(default=1, validator=instance_of(int))
    laser_freq         = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth           = attr.ib(default=800e-12, validator=instance_of(float))
    use_tag_bits       = attr.ib(default=False, validator=instance_of(bool))
    use_sweeps         = attr.ib(default=False, validator=instance_of(bool))
    time_after_sweep   = attr.ib(default=int(96), validator=instance_of(int))
    acq_delay          = attr.ib(default=int(0), validator=instance_of(int))
    line_freq          = attr.ib(default=7900.0, validator=instance_of(float))
    line_delta         = attr.ib(init=False)
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
    def total_sweep_time(self):
        return self.acq_delay + self.data_range + self.time_after_sweep

    def __allocate_data_by_channel(self, df: pd.DataFrame):
        """
        Go over the channels and find the events from that specific channel, assigning
        them to a dictionary with a suitable name.
        :param df: DataFrame with data to allocate.
        :return: Dict containing the data
        """
        dict_of_data = {}
        self.data_to_grab = ['abs_time', 'edge', 'sweep', 'tag']

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
                                                            cols_in_data=self.data_to_grab, use_sweeps=self.use_sweeps,
                                                            max_sweep=df['sweep'].max(),
                                                            total_sweep_time=self.total_sweep_time)
        dict_of_data = validate_frame_input(dict_of_data=dict_of_data,
                                            num_of_lines=self.y_pixels, binwidth=self.binwidth,
                                            last_event_time=last_event_time,
                                            cols_in_data=self.data_to_grab)
        try:
            dict_of_data['Laser'] = validate_laser_input(dict_of_data['Laser'], laser_freq=self.laser_freq,
                                                         binwidth=self.binwidth)
        except KeyError:
            pass

        validate_created_data_channels(dict_of_data)
        return dict_of_data

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

        if len(self.data) == 0:
            raise IOError('List file contained zero events.')

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
