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
            "1a": 44,
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


def get_range(filename: str = '') -> int:
    """
    Finds the "range" of the current file in the proper units
    :return: range as defined my MCS, after bit depth multiplication
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    format_range = re.compile(r'range=(\d+)')
    with open(filename, 'r') as f:
        cur_str = f.read(500)

    range_before_bit_depth = int(re.search(format_range, cur_str).group(1))
    format_bit_depth = re.compile(r'bitshift=(\w+)')
    bit_depth_wrong_base = re.search(format_bit_depth, cur_str).group(1)
    bit_depth_as_hex = bit_depth_wrong_base[-2:]  # last 2 numbers count
    range_after_bit_depth = range_before_bit_depth * 2 ** int(bit_depth_as_hex, 16)

    assert isinstance(range_after_bit_depth, int)
    return range_after_bit_depth


def get_timepatch(filename: str = '') -> str:
    """
    Get the time patch value out of of a list file.
    :param filename: File to be read.
    :return: Time patch value as string.
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    format_timepatch = re.compile(r'time_patch=(\w+)')
    with open(filename, 'r') as f:
        cur_str = f.read(5000)  # read 5000 chars for the timepatch value

    timepatch = re.search(format_timepatch, cur_str).group(1)
    # data_length_dict = create_data_length_dict()
    # data_length = data_length_dict[timepatch]

    assert isinstance(timepatch, str)
    # assert isinstance(data_length, int)
    # return timepatch, data_length - DEPRECATED
    return timepatch


def find_active_channels(filename: str = '') -> List:
    """
    Create a dictionary containing the active channels.
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    format_active = re.compile(r'active=(\d)')
    active_channels = [False, False, False, False, False, False]

    with open(filename, 'r') as f:
        cur_str = f.read(5000)

    list_of_matches = re.findall(format_active, cur_str)

    for idx, match in enumerate(list_of_matches):
        if match == '1':
            active_channels[idx] = True

    return active_channels


def get_start_pos(filename: str = '') -> int:
    """
    Returns the start position of the data
    :param filename: Name of file
    :return: Integer of file position for f.seek() method
    """
    import re

    if filename == '':
        raise ValueError('No filename given.')

    format_data = re.compile(r"DATA]\n")
    pos_in_file = 0
    with open(filename, 'r') as f:
        while pos_in_file == 0:
            line = f.readline()
            match = re.search(format_data, line)
            if match is not None:
                pos_in_file = f.tell()
                return pos_in_file  # to have the [DATA] as header


def read_lst_file_debug(filename: str='', start_of_data_pos: int=0, num_of_lines=0) -> pd.DataFrame:
    """
    Read the list file into a dataframe.
    :param filename: Name of list file.
    :param start_of_data_pos: The place in chars in the file that the data starts at.
    :return: Dataframe with all events registered.
    """
    from itertools import islice

    if filename is '' or start_of_data_pos == 0:
        return ValueError('Wrong input detected.')

    with open(filename, 'r') as f:
        f.seek(start_of_data_pos)
        n_lines = list(islice(f, int(num_of_lines)))
        if not n_lines:
            pass
        file_separated = [st.rstrip() for st in n_lines]
    df = pd.DataFrame(file_separated, columns=['raw'], dtype=str)

    assert df.shape[0] > 0
    return df


def read_lst_file(filename: str = '', start_of_data_pos: int = 0) -> pd.DataFrame:
    """
    Read the list file into a dataframe.
    :param filename: Name of list file.
    :param start_of_data_pos: The place in chars in the file that the data starts at.
    :return: Dataframe with all events registered.
    """
    # TRIAL VERSION, DEPRECATED. TRY TO RAISE FROM DEAD ONLY IF CURRENT VERSION IS SLOW ###############
    # def binarize(str1):
    #     return "{0:0{1}b}".format(int(str1, 16), data_length)
    #
    # with open(filename, 'r') as f:
    #     dict_bin = dict([('bin', binarize)])
    #     f.seek(start_of_data_pos)
    #     df = pd.read_csv(f, header=0, converters=dict_bin, dtype=str, names=['bin'])
    ######################################################################################################

    if filename is '' or start_of_data_pos == 0:
        return ValueError('Wrong input detected.')

    # with open(filename, 'r') as f:
    #     f.seek(start_of_data_pos)
    #     file_separated = [line.rstrip() for line in f]  # TODO: SciLuigi?
    # df = pd.DataFrame(file_separated, columns=['raw'], dtype=str)

    with open(filename, "rb") as f:
        f.seek(start_of_data_pos)
        arr = np.fromfile(f, dtype='14S').astype('14U')
        arr1 = np.core.defchararray.rstrip(arr, "\r\n")

    df = pd.DataFrame(arr1, columns=['raw'], dtype=str)

    assert df.shape[0] > 0
    return df


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
    if lst_input.count(True) != len(user_inputs):
        raise UserWarning('Wrong number of user inputs ({}) compared to number of actual inputs ({}).'.
                          format(len(user_inputs), lst_input.count(True)))

    help_dict = {
        '001': 0,
        '010': 1,
        '110': 2
    }

    for key in user_inputs:
        assert lst_input[help_dict[user_inputs[key]]] is True
