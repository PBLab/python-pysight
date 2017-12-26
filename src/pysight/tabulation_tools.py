"""
__author__ = Hagai Hargil
"""
import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple, List
import warnings
from pysight.apply_df_funcs import get_lost_bit_np, get_lost_bit_tag, iter_string_hex_to_bin, convert_hex_to_int, convert_hex_to_bin
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
    data_range         = attr.ib(default=1, validator=instance_of(int))
    is_binary          = attr.ib(default=False, validator=instance_of(bool))
    use_tag_bits       = attr.ib(default=False, validator=instance_of(bool))
    time_after_sweep   = attr.ib(default=int(96), validator=instance_of(int))
    acq_delay          = attr.ib(default=int(0), validator=instance_of(int))
    num_of_channels    = attr.ib(default=3, validator=instance_of(int))
    df_after_timepatch = attr.ib(init=False)


    def run(self):
        """ Pipeline of analysis """
        if self.is_binary:
            self.df_after_timepatch = self.tabulate_input_binary()
        else:
            self.df_after_timepatch = self.tabulate_input_hex()
        print('Sorted dataframe created. Starting setting the proper data channel distribution...')

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

    def process_chan_edge(self, struct_of_data) -> Tuple[np.ndarray, np.ndarray]:
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
            thrown_channels = 0
            keys_to_pop = []
            for key, item in self.dict_of_inputs.items():
                if item not in actual_data_channels:
                    keys_to_pop.append(key)
                    thrown_channels += 1
            self.num_of_channels -= thrown_channels
            [self.dict_of_inputs.pop(key) for key in keys_to_pop]
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
