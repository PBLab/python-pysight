"""
__author__ = Hagai Hargil
"""
import pandas as pd
from typing import Dict, List
import numpy as np



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


def determine_binary(filename: str = '') -> bool:
    """
    Determine whether we're dealing with a binary or a hex file.
    :param filename: File name
    :return: Boolean value
    """
    if filename == '':
        raise ValueError('No filename given.')

    try:
        with open(filename, 'r') as f:
            f.read(100)
            is_binary = False
    except UnicodeDecodeError:  # File is binary
        with open(filename, 'rb') as f:
            f.read(100)
            is_binary = True
    else:
        raise ValueError('File read unsuccessfully.')
    finally:
        return is_binary


def get_range(filename: str = '', is_binary: bool = False) -> int:
    """
    Finds the "range" of the current file in the proper units
    :return: range as defined my MCS, after bit depth multiplication
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    if is_binary:
        format_str = b'range=(\d+)'
        file_mode = 'rb'
    else:
        format_str = r'range=(\d+)'
        file_mode = 'r'

    format_range = re.compile(format_str)
    with open(filename, file_mode) as f:
        cur_str = f.read(500)

    range = int(re.search(format_range, cur_str).group(1))

    return range


def get_timepatch(filename: str = '', is_binary: bool = False) -> str:
    """
    Get the time patch value out of of a list file.
    :param filename: File to be read.
    :return: Time patch value as string.
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    if is_binary:
        format_str = b'time_patch=(\w+)'
        file_mode = 'rb'
    else:
        format_str = r'time_patch=(\w+)'
        file_mode = 'r'

    format_timepatch = re.compile(format_str)
    with open(filename, file_mode) as f:
        cur_str = f.read(5000)  # read 5000 chars for the timepatch value

    timepatch = re.search(format_timepatch, cur_str).group(1)
    try:
        timepatch = timepatch.decode('utf-8')
    except AttributeError:
        assert isinstance(timepatch, str)
    finally:
        return timepatch


def find_active_channels(filename: str = '', is_binary: bool = False) -> List:
    """
    Create a dictionary containing the active channels.
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    if is_binary:
        format_str = b'active=(\d)'
        file_mode = 'rb'
        match = b'1'
    else:
        format_str = r'active=(\d)'
        file_mode = 'r'
        match = '1'

    format_active = re.compile(format_str)
    active_channels = [False, False, False, False, False, False]

    with open(filename, file_mode) as f:
        cur_str = f.read(5000)

    list_of_matches = re.findall(format_active, cur_str)

    for idx, cur_match in enumerate(list_of_matches):
        if cur_match == match:
            active_channels[idx] = True

    return active_channels


def get_start_pos(filename: str = '', is_binary: bool = False) -> int:
    """
    Returns the start position of the data
    :param filename: Name of file
    :return: Integer of file position for f.seek() method
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    if is_binary:
        format_str = b'DATA]\r\n'
        file_mode = 'rb'
    else:
        format_str = r'DATA]\n'
        file_mode = 'r'

    format_data = re.compile(format_str)
    pos_in_file = 0
    with open(filename, file_mode) as f:
        while pos_in_file == 0:
            line = f.readline()
            match = re.search(format_data, line)
            if match is not None:
                pos_in_file = f.tell()
                return pos_in_file  # to have the [DATA] as header


def read_lst(filename: str = '', start_of_data_pos: int = 0, timepatch: str = '',
             num_of_items=-1, is_binary: bool = False) -> np.array:
    """
    Updated version of LST readout using array slicing (and not Pandas slicing).
    :param filename: File to read.
    :param start_of_data_pos: Index of first data word.
    :param timepatch:
    :param num_of_items: Number of lines to read. -1 is all file.
    :param is_binary: Determines whether file is binary or hex.
    :return:
    """

    if filename is '' or start_of_data_pos == 0 or timepatch == '':
        return ValueError('Wrong input detected.')

    data_length_dict = create_data_length_dict()
    # Make sure we read the exact number of lines we were asked to
    if is_binary:
        data_length = data_length_dict[timepatch] // 8
    else:
        data_length = data_length_dict[timepatch] // 4 + 2

    if num_of_items == -1:
        num_of_lines_to_read = -1
    else:
        num_of_lines_to_read = int(num_of_items * data_length)

    # Read file
    if is_binary:
        with open(filename, "rb") as f:
            f.seek(start_of_data_pos)
            dt = np.dtype(('u1', data_length))
            arr = np.fromfile(f, dtype=dt, count=num_of_lines_to_read)
            arr_as_bits = np.unpackbits(np.fliplr(arr), axis=1)
            ########## STILL NOT SUPPORTED ##############
            raise NotImplementedError('Binary files are still not supported.')
        return arr_as_bits

    else:
        with open(filename, "rb") as f:
            f.seek(start_of_data_pos)
            arr = np.fromfile(f, dtype='{}S'.format(data_length),
                              count=num_of_lines_to_read).astype('{}U'.format(data_length))

        return arr


def create_inputs_dict(gui=None) -> Dict:
    """
    Create a dictionary for all input channels. Currently allows for three channels.
    'Empty' channels will not be checked.
    """

    if gui is None:
        raise ValueError('No GUI received.')

    dict_of_inputs = {}

    if gui.input_start.get() != 'Empty':
        dict_of_inputs[gui.input_start.get()] = '110'

    if gui.input_stop1.get() != 'Empty':
        dict_of_inputs[gui.input_stop1.get()] = '001'

    if gui.input_stop2.get() != 'Empty':
        dict_of_inputs[gui.input_stop2.get()] = '010'

    assert len(dict_of_inputs) >= 1
    assert 'Empty' not in list(dict_of_inputs.keys())

    return dict_of_inputs


def compare_recorded_and_input_channels(user_inputs: Dict, lst_input: List):
    """
    Raise error if user gave wrong amount of inputs
    :param user_inputs: Dict of user inputs
    :param lst_input: Actual recorded data from multiscaler
    """

    # If a user assumes an input exists, but it doesn't - raise an error
    if lst_input.count(True) < len(user_inputs):
        raise UserWarning('Wrong number of user inputs ({}) compared to number of actual inputs ({}) to the multiscaler.'
                          .format(len(user_inputs), lst_input.count(True)))

    help_dict = {
        '001': 0,
        '010': 1,
        '110': 2
    }

    for key in user_inputs:
        if not lst_input[help_dict[user_inputs[key]]]:
            raise UserWarning('Wrong channel specification - the key {} is on an empty channel (number {}).'.\
                  format(key, user_inputs[key]))

