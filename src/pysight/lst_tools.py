import pandas as pd
import numpy as np
from typing import Dict
from collections import OrderedDict
from numba import jit, int64, uint64
import warnings
from pysight.apply_df_funcs import get_lost_bit_np, iter_string_hex_to_bin, convert_hex_to_int
from pysight.validation_tools import validate_line_input, validate_frame_input, \
    validate_laser_input, validate_created_data_channels


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


@jit(nopython=True, cache=True)
def numba_sorted(arr: np.array) -> np.array:
    """
    Sort an array with Numba. CURRENTLY NOT WORKING
    """

    arr.sort()
    return arr.astype(np.uint64)


def determine_data_channels(df: pd.DataFrame=None, dict_of_inputs: Dict=None,
                            num_of_frames: int=-1, x_pixels: int=-1, y_pixels: int=-1,
                            laser_freq: float=80e6, binwidth: float=800e-12,
                            flyback: float=0.001) -> Dict:
    """ Create a dictionary that contains the data in its ordered form."""

    if df.empty:
        raise ValueError('Received dataframe was empty.')

    dict_of_data = {}
    for key in dict_of_inputs:
        relevant_values = df.loc[df['channel'] == dict_of_inputs[key], 'abs_time']
        # NUMBA SORT NOT WORKING:
        # sorted_vals = numba_sorted(relevant_values.values)
        # dict_of_data[key] = pd.DataFrame(sorted_vals, columns=['abs_time'])
        if 'PMT1' == key or 'PMT2' == key:
            dict_of_data[key] = relevant_values.reset_index(drop=True)
        else:
            dict_of_data[key] = relevant_values.sort_values().reset_index(drop=True)

    # Validations
    dict_of_data, line_delta = validate_line_input(dict_of_data=dict_of_data, num_of_lines=y_pixels,
                                                   num_of_frames=num_of_frames, binwidth=binwidth)
    dict_of_data = validate_frame_input(dict_of_data=dict_of_data, flyback=flyback, binwidth=binwidth,
                                        line_delta=line_delta, num_of_lines=y_pixels)
    try:
        dict_of_data['Laser'] = validate_laser_input(dict_of_data['Laser'], laser_freq=laser_freq, binwidth=binwidth)
    except KeyError:
        pass

    validate_created_data_channels(dict_of_data)
    return dict_of_data


@jit((int64[:](uint64[:], uint64[:])), nopython=True, cache=True)
def numba_search_sorted(input_sorted, input_values):
    """ Numba-powered searchsorted function. """
    return np.searchsorted(input_sorted, input_values) - 1


def allocate_photons(dict_of_data=None, gui=None) -> pd.DataFrame:
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
    df_photons = pd.DataFrame(df_photons, dtype=np.uint64)  # before this change it's a series with a name, not column head
    column_heads = {'Lines': 'time_rel_line', 'Frames': 'time_rel_frames', 'Laser': 'time_rel_pulse'}

    # Main loop - Sort lines and frames for all photons and calculate relative time
    for key in relevant_keys:
        sorted_indices = numba_search_sorted(dict_of_data[key].values, df_photons['abs_time'].values)
        try:
            df_photons[key] = dict_of_data[key].loc[sorted_indices].values
        except KeyError:
            warnings.warn('All computed sorted_indices were "-1" for key {}. Trying to resume...'.format(key))

        df_photons.dropna(how='any', inplace=True)
        df_photons[key] = df_photons[key].astype(np.uint64)
        df_photons[column_heads[key]] = df_photons['abs_time'] - df_photons[key]  # relative time of each photon in
        # accordance to the line\frame\laser pulse
        # TODO: Remove photons that are detected during the "turn-around" of the resonant mirror
        if key != 'Laser':
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
        df_photons = interpolate_tag(df_photons=df_photons, tag_data=tag, gui=gui)
        print('TAG lens interpolation finished.')

    df_photons.drop(['abs_time'], axis=1, inplace=True)

    return df_photons


def process_chan_edge(struct_of_data):
    """
    Simple processing scheme for the channel and edge data.
    """
    bin_array = np.array(iter_string_hex_to_bin("".join(struct_of_data.data)))
    edge = slice_string_arrays(bin_array, start=0, end=1)
    channel = slice_string_arrays(bin_array, start=1, end=4)

    return edge, channel


def tabulate_input_hex(data: np.array, dict_of_slices: OrderedDict, data_range: int, input_channels: Dict) -> pd.DataFrame:
    """
    Reformat the read hex data into a dataframe.
    """

    for key in list(dict_of_slices.keys())[1:]:
        dict_of_slices[key].data = slice_string_arrays(data, dict_of_slices[key].start,
                                                       dict_of_slices[key].end)

    # Channel and edge information
    edge, channel = process_chan_edge(dict_of_slices.pop('chan_edge'))
    # TODO: Timepatch == '3' is not supported because of this loop.

    if dict_of_slices['lost'] is True:
        for key in list(dict_of_slices.keys())[1:]:
            if dict_of_slices[key].needs_bits:
                list_with_lost = iter_string_hex_to_bin("".join(dict_of_slices[key].data))
                step_size = dict_of_slices[key].end - dict_of_slices[key].start
                list_of_losts, dict_of_slices[key].processed = get_lost_bit_np(list_with_lost, step_size, len(data))
            else:
                dict_of_slices[key].processed = convert_hex_to_int(dict_of_slices[key].data)
    else:
        for key in list(dict_of_slices.keys())[1:]:
            dict_of_slices[key].processed = convert_hex_to_int(dict_of_slices[key].data)

    # Reformat data
    df = pd.DataFrame(channel, columns=['channel'], dtype='category')
    df['edge'] = edge
    df['edge'] = df['edge'].astype('category')

    try:
        df['tag'] = dict_of_slices['tag'].processed
    except KeyError:
        pass

    try:
        df['lost'] = list_of_losts  # TODO: Currently the LOST bit is meaningless
        df['lost'] = df['lost'].astype('category')
    except NameError:
        pass

    df['abs_time'] = np.uint64(0)

    if 'sweep' in dict_of_slices:
        df['abs_time'] = dict_of_slices['abs_time'].processed + (dict_of_slices['sweep'].processed - 1) * data_range
    else:
        df['abs_time'] = dict_of_slices['abs_time'].processed

    # Before sorting all photons make sure that no input is missing from the user. If it's missing
    # the code will ignore this channel, but not raise an exception
    actual_data_channels = set(df['channel'].cat.categories.values)
    if actual_data_channels != set(input_channels.values()):
        warnings.warn("Channels that were inserted in GUI don't match actual data channels recorded. \n"
                      "The list files contains data in the following channels: {}.".format(actual_data_channels))

    assert np.all(df['abs_time'].values >= 0)

    return df


def slice_string_arrays(arr: np.array, start: int, end: int) -> np.array:
    """
    Slice an array of strings efficiently.
    Based on http://stackoverflow.com/questions/39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
    with modifications for Python 3.
    """
    b = arr.view('U1').reshape(len(arr), -1)[:, start:end]
    return np.fromstring(b.tostring(), dtype='U' + str(end - start))


def tabulate_input_binary(data: np.array, dict_of_slices: OrderedDict, data_range: int, input_channels: Dict) -> pd.DataFrame:
    """
    Reformat the read binary data into a dataframe.
    """
    num_of_lines = data.shape[0]

    for key in dict_of_slices:
        cur_data = data[:, dict_of_slices[key].start:dict_of_slices[key].end]
        try:
            zero_arr = np.zeros(num_of_lines, dict_of_slices[key].cols)

        except AttributeError:  # No cols field since number of bits is a multiple of 8
            dict_of_slices[key].data_as_

