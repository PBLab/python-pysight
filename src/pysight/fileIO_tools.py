"""
__author__ = Hagai Hargil
"""
import pandas as pd
from typing import Dict, List
import numpy as np
import attr


@attr.s(slots=True)
class FileIO(object):
    """
    Manage pipeline of file IO process
    """
    filename    = attr.ib(validator=attr.validators.instance_of(str))
    debug       = attr.ib(validator=attr.validators.instance_of(int))
    input_start = attr.ib(validator=attr.validators.instance_of(str))
    input_stop1 = attr.ib(validator=attr.validators.instance_of(str))
    input_stop2 = attr.ib(validator=attr.validators.instance_of(str))
    is_binary   = attr.ib(init=False)
    timepatch   = attr.ib(init=False)
    data_range  = attr.ib(init=False)
    start_of_data_pos = attr.ib(init=False)
    dict_of_input_channels = attr.ib(init=False)
    list_of_recorded_data_channels = attr.ib(init=False)
    data = attr.ib(init=False)

    def run(self):
        # Open file and find the needed parameters
        self.is_binary = self.determine_binary()
        self.timepatch = self.get_timepatch()
        if self.timepatch == '3' and not self.is_binary:
            raise NotImplementedError(
                'Timepatch value "3" is currently not supported for hex files. Please message the package owner.')
        self.data_range = self.get_range()

        self.start_of_data_pos = self.get_start_pos()
        self.dict_of_input_channels = self.create_inputs_dict()
        self.list_of_recorded_data_channels = self.find_active_channels()
        self.compare_recorded_and_input_channels()
        num_of_items = self.determine_num_of_items()
        self.data = self.read_lst(num_of_items=num_of_items)
        print('File read. Sorting the file according to timepatch...')

    def determine_num_of_items(self):
        if self.debug == 0:
            num_of_items = -1
            read_string = 'Reading file "{}"...'.format(self.filename)
        else:
            num_of_items = 0.5e6
            read_string = '[DEBUG] Reading file "{}"...'.format(self.filename)

        print(read_string)
        return num_of_items

    @staticmethod
    def create_data_length_dict():
        """
        CURRENTLY DEPRECATED
        :return:
        """
        dict_of_data_length = {
                "0": 16,
                "5": 32,
                "1": 32,
                "1a": 48,
                "2a": 48,
                "22": 48,
                "32": 48,
                "2": 48,
                "5b": 64,
                "Db": 64,
                "f3": 64,
                "43": 64,
                "c3": 64,
                "3": 64
            }

        return dict_of_data_length

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

    def determine_binary(self) -> bool:
        """
        Determine whether we're dealing with a binary or a hex file.
        :return: Boolean value
        """
        if self.filename == '':
            raise ValueError('No filename given.')

        try:
            with open(self.filename, 'r') as f:
                f.read(100)
                is_binary = False
        except UnicodeDecodeError:  # File is binary
            with open(self.filename, 'rb') as f:
                f.read(100)
                is_binary = True
        except FileNotFoundError:
            raise FileNotFoundError("File {} doesn't exist.".format(self.filename))

        else:
            raise ValueError('File [} read unsuccessfully.'.format(self.filename))
        finally:
            return is_binary

    def get_range(self) -> int:
        """
        Finds the "range" of the current file in the proper units
        :return: range as defined my MCS, after bit depth multiplication
        """
        import re

        if self.filename == '':
            raise ValueError('No filename given.')

        if self.is_binary:
            format_str = b'range=(\d+)'
            file_mode = 'rb'
        else:
            format_str = r'range=(\d+)'
            file_mode = 'r'

        format_range = re.compile(format_str)
        with open(self.filename, file_mode) as f:
            cur_str = f.read(500)

        range = int(re.search(format_range, cur_str).group(1))

        return range

    def get_timepatch(self) -> str:
        """
        Get the time patch value out of of a list file.
        :param filename: File to be read.
        :return: Time patch value as string.
        """
        import re

        if self.filename == '':
            raise ValueError('No filename given.')

        if self.is_binary:
            format_str: str = b'time_patch=(\w+)'
            file_mode: str = 'rb'
        else:
            format_str: str = r'time_patch=(\w+)'
            file_mode: str = 'r'

        format_timepatch = re.compile(format_str)
        with open(self.filename, file_mode) as f:
            cur_str: str = f.read(5000)  # read 5000 chars for the timepatch value

        timepatch = re.search(format_timepatch, cur_str).group(1)
        try:
            timepatch = timepatch.decode('utf-8')
        except AttributeError:
            assert isinstance(timepatch, str)
        finally:
            return timepatch

    def find_active_channels(self) -> List[bool]:
        """
        Create a dictionary containing the active channels.
        """
        import re

        if self.filename == '':
            raise ValueError('No filename given.')

        if self.is_binary:
            format_str = b'active=(\d)'
            file_mode = 'rb'
            match = b'1'
        else:
            format_str = r'active=(\d)'
            file_mode = 'r'
            match = '1'

        format_active = re.compile(format_str)
        active_channels: List[bool] = [False, False, False, False, False, False]

        with open(self.filename, file_mode) as f:
            cur_str: str = f.read(5000)

        list_of_matches = re.findall(format_active, cur_str)

        for idx, cur_match in enumerate(list_of_matches):
            if cur_match == match:
                active_channels[idx] = True

        return active_channels

    def get_start_pos(self) -> int:
        """
        Returns the start position of the data
        :param filename: Name of file
        :return: Integer of file position for f.seek() method
        """
        import re

        if self.filename == '':
            raise ValueError('No filename given.')

        if self.is_binary:
            format_str = b'DATA]\r\n'
            file_mode = 'rb'
        else:
            format_str = r'DATA]\n'
            file_mode = 'r'

        format_data = re.compile(format_str)
        pos_in_file: int = 0
        with open(self.filename, file_mode) as f:
            while pos_in_file == 0:
                line = f.readline()
                match = re.search(format_data, line)
                if match is not None:
                    pos_in_file = f.tell()
                    return pos_in_file  # to have the [DATA] as header

    def read_lst(self, num_of_items: int=-1) -> np.ndarray:
        """
        Updated version of LST readout using array slicing (and not Pandas slicing).
        :param num_of_items: Number of lines to read. -1 is all file.
        :return:
        """

        if self.filename is '' or self.start_of_data_pos == 0 or self.timepatch == '':
            return ValueError('Wrong input detected.')

        data_length_dict = self.create_data_length_dict()
        # Make sure we read the exact number of lines we were asked to
        if self.is_binary:
            data_length = data_length_dict[self.timepatch] // 8
        else:
            data_length = data_length_dict[self.timepatch] // 4 + 2

        if num_of_items == -1:
            num_of_lines_to_read = -1
        else:
            num_of_lines_to_read = int(num_of_items * data_length)

        # Read file
        if self.is_binary:
            with open(self.filename, "rb") as f:
                f.seek(self.start_of_data_pos)
                dt: np.dtype = np.dtype(('u1', data_length))
                arr: np.ndarray = np.fromfile(f, dtype=dt, count=num_of_lines_to_read)
                arr_as_bits: np.ndarray = np.unpackbits(np.fliplr(arr), axis=1)
                ########## STILL NOT SUPPORTED ##############
                # It's very hard to pack int32 bits, for example, to a numpy array effectively,
                # since numpy.packbits only works for uint8. Other solutions seem slow and complicated.
                raise NotImplementedError('Binary files are still not supported.')
            return arr_as_bits

        else:
            with open(self.filename, "rb") as f:
                f.seek(self.start_of_data_pos)
                arr = np.fromfile(f, dtype='{}S'.format(data_length),
                                  count=num_of_lines_to_read).astype('{}U'.format(data_length))

            return arr

    def create_inputs_dict(self) -> Dict[str, str]:
        """
        Create a dictionary for all input channels. Currently allows for three channels.
        'Empty' channels will not be checked.
        """
        dict_of_inputs = {}

        if self.input_start != 'Empty':
            dict_of_inputs[self.input_start] = '110'

        if self.input_stop1 != 'Empty':
            dict_of_inputs[self.input_stop1] = '001'

        if self.input_stop2 != 'Empty':
            dict_of_inputs[self.input_stop2] = '010'

        assert len(dict_of_inputs) >= 1
        assert 'Empty' not in list(dict_of_inputs.keys())

        return dict_of_inputs

    def compare_recorded_and_input_channels(self) -> None:
        """
        Raise error if user gave wrong amount of inputs
        """

        # If a user assumes an input exists, but it doesn't - raise an error
        if self.list_of_recorded_data_channels.count(True) < len(self.dict_of_input_channels):
            raise UserWarning('Wrong number of user inputs ({}) compared to number of actual inputs ({}) to the multiscaler.'
                              .format(len(self.dict_of_input_channels), self.list_of_recorded_data_channels.count(True)))

        help_dict = {
            '001': 0,
            '010': 1,
            '110': 2
        }

        for key in self.dict_of_input_channels:
            if not self.list_of_recorded_data_channels[help_dict[self.dict_of_input_channels[key]]]:
                raise UserWarning('Wrong channel specification - the key {} is on an empty channel (number {}).'.\
                      format(key, self.dict_of_input_channels[key]))
