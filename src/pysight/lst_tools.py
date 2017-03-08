import pandas as pd
import numpy as np
from typing import Dict
from numba import jit
import warnings


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


def create_data_length_dict():
    """
    CURRENTLY DEPRECATED
    :return:
    """
    dict_of_data_length = \
        {
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
        cur_str = f.read(500)  # read 100 chars for the basic values

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

    with open(filename, 'r') as f:
        f.seek(start_of_data_pos)
        file_separated = [line.rstrip() for line in f]  # TODO: SciLuigi?
        # TODO: f = io.open(r'C:\Users\Hagai\Documents\GitHub\python-pysight\multiscalerAsPreampMicrogliaMouseFLIM035.lst', 'rb')
        # f.seek(1590)
        # numread = 0
        #
        # for i in itertools.count():
        #     numread = f.readinto(b)
        #     lista.append(b.decode())
        #     if not numread:
        #         break
        # array.array can also work, with fromfile()
    df = pd.DataFrame(file_separated, columns=['raw'], dtype=str)

    assert df.shape[0] > 0
    return df


def timepatch_sort(df, timepatch: str='', data_range: int=0, input_channels: Dict=None) -> pd.DataFrame:
    """
    Takes a raw dataframe and sorts it to columns according to its timepatch value.
    :param df: Input DF.
    :param timepatch: Key by which we sort.
    :param data_range: Data range of file.
    :param input_channels: dictionary of actual input channels
    """
    from pysight import timepatch_manager

    # Verify inputs
    if df.shape[0] == 0 or timepatch == '' or data_range == 0 or input_channels is None:
        raise ValueError("Wrong inputs inserted.")

    # Create dictionary for hex to bin conversion
    hex_to_bin = hex_to_bin_dict()

    # %% Analyze channel and edge information
    df['bin'] = df['raw'].str[-1].map(hex_to_bin)
    df['channel'] = df['bin'].str[-3:].astype(dtype='category')
    df['edge'] = df['bin'].str[-4].astype(dtype='category')

    # Before sorting all photons make sure that no input is missing from the user.
    actual_data_channels = set(df['channel'].cat.categories.values)
    if actual_data_channels != set(input_channels.values()):
        raise UserWarning("Channels that were inserted in GUI don't match actual data channels recorded. \n"
                          "Recorded channels are {}.".format(actual_data_channels))

    # Start going through the df and extract the bits
    df['abs_time'] = np.uint64(0)
    df_after_timepatch = timepatch_manager.ChoiceManager().process(timepatch, data_range, df)

    df_after_timepatch.drop(['bin', 'raw', 'abs_time_as_str'], axis=1, inplace=True)
    try:
        df_after_timepatch.drop(['tag_as_str'], axis=1, inplace=True)
    except ValueError:
        pass
    try:
        df_after_timepatch.drop(['sweep_as_str'], axis=1, inplace=True)
    except ValueError:
        pass

    if list(df_after_timepatch.columns) != ['channel', 'edge', 'abs_time', 'sweep', 'tag', 'lost']:
        raise ValueError('Wrong dataframe created.')

    assert np.all(df_after_timepatch['abs_time'].values >= 0)
    return df_after_timepatch


def create_frame_array(lines: pd.Series=None, last_event_time: int=None,
                       pixels: int=None, spacing_between_lines: int=None) -> np.ndarray:
    """Create a pandas Series of start-of-frame times"""

    if last_event_time is None or pixels is None or lines.empty:
        raise ValueError('Wrong input detected.')

    if last_event_time <= 0:
        raise ValueError('Last event time is zero or negative.')

    num_of_recorded_lines = lines.shape[0]
    actual_num_of_frames = num_of_recorded_lines // pixels
    unnecess_lines = actual_num_of_frames % pixels
    if unnecess_lines == 0:
        array_of_frames = np.linspace(start=0, stop=last_event_time, num=int(actual_num_of_frames), endpoint=False)
    else:
        last_event_time = lines[num_of_recorded_lines - unnecess_lines] + spacing_between_lines
        array_of_frames = np.linspace(start=0, stop=last_event_time, num=int(actual_num_of_frames), endpoint=False)

    return array_of_frames


def create_line_array(last_event_time: int=None, num_of_lines=None, num_of_frames=None) -> np.ndarray:
    """Create a pandas Series of start-of-line times"""

    if (last_event_time is None) or (num_of_lines is None) or (num_of_frames is None):
        raise ValueError('Wrong input detected.')

    if (num_of_lines <= 0) or (num_of_frames <= 0):
        raise ValueError('Number of lines and frames has to be positive.')

    if last_event_time <= 0:
        raise ValueError('Last event time is zero or negative.')

    total_lines = num_of_lines * int(num_of_frames)
    line_array = np.linspace(start=0, stop=last_event_time, num=total_lines)
    return line_array


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

def validate_created_data_channels(dict_of_data: Dict):
    """
    Make sure that the dictionary that contains all data channels makes sense.
    """
    assert {'PMT1', 'Lines', 'Frames'} <= set(dict_of_data.keys())  # A is subset of B

    if dict_of_data['Frames'].shape[0] > dict_of_data['Lines'].shape[0]:  # more frames than lines
        raise UserWarning('More frames than lines, replace the two.')

    try:
        if dict_of_data['TAG Lens'].shape[0] < dict_of_data['Lines'].shape[0]:
            raise UserWarning('More lines than TAG pulses, replace the two.')
    except KeyError:
        pass

    try:
        if dict_of_data['Laser'].shape[0] < dict_of_data['Lines'].shape[0] or \
           dict_of_data['Laser'].shape[0] < dict_of_data['Frames'].shape[0]:
            raise UserWarning('Laser pulses channel contained less ticks than the Lines or Frames channel.')
    except KeyError:
        pass

    try:
        if dict_of_data['Laser'].shape[0] < dict_of_data['TAG Lens'].shape[0]:
            raise UserWarning('Laser pulses channel contained less ticks than the TAG lens channel.')
    except KeyError:
        pass



def determine_data_channels(df: pd.DataFrame=None, dict_of_inputs: Dict=None,
                            num_of_frames: int=-1, x_pixels: int=-1, y_pixels: int=-1) -> Dict:
    """ Create a dictionary that contains the data in its ordered form."""

    if df.empty:
        raise ValueError('Received dataframe was empty.')

    dict_of_data = {}
    for key in dict_of_inputs:
        dict_of_data[key] = df.loc[df['channel'] == dict_of_inputs[key], 'abs_time'].reset_index(drop=True)
        # TODO: GroupBy the line above?
    if 'Lines' not in dict_of_data.keys():  # A 'Lines' channel has to exist to create frames
        last_event_time = dict_of_data['PMT1'].max()  # Assuming only data from PMT1 is relevant here
        line_array = create_line_array(last_event_time=last_event_time, num_of_lines=y_pixels,
                                       num_of_frames=num_of_frames)
        dict_of_data['Lines'] = pd.Series(line_array, name='abs_time', dtype=np.uint64)

    if 'Frames' not in dict_of_data.keys():  # A 'Frames' channel has to exist to create frames
        spacing_between_lines = np.abs(dict_of_data['Lines'].diff()).mean()
        last_event_time = int(dict_of_data['PMT1'].max() + spacing_between_lines)  # Assuming only data from PMT1 is relevant here
        frame_array = create_frame_array(lines=dict_of_data['Lines'], last_event_time=last_event_time,
                                         pixels=x_pixels, spacing_between_lines=spacing_between_lines)
        dict_of_data['Frames'] = pd.Series(frame_array, name='abs_time', dtype=np.uint64)
    else:  # Add 0 to the first entry of the series
        dict_of_data['Frames'] = pd.Series([0], name='abs_time').append(dict_of_data['Frames'], ignore_index=True)

    # Validations
    validate_created_data_channels(dict_of_data)

    return dict_of_data


@jit(nopython=True, cache=True)
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
