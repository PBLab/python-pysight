"""
__author__ = Hagai Hargil
"""
from typing import Dict
import pandas as pd
import numpy as np
import warnings


def validate_line_input(dict_of_data: Dict, num_of_lines: int = -1, num_of_frames: int = -1, binwidth: float = 800e-12):
    """ Verify that the .lst input of lines exists and looks fine. Create one if there's no such input. """
    if num_of_lines == -1:
        raise ValueError('No number of lines input received.')

    if num_of_frames == -1:
        raise ValueError('No number of frames received.')

    last_event_time = dict_of_data['PMT1'].max()  # TODO: Assuming only data from PMT1 is relevant here
    if 'Lines' in dict_of_data.keys():
        # Verify that the input is not corrupt
        median_of_lines = dict_of_data['Lines'].diff().abs().median()
        mean_of_lines = dict_of_data['Lines'].diff().abs().mean()
        diff_between_mean_median = abs((mean_of_lines - median_of_lines)/median_of_lines)
        if diff_between_mean_median > 0.15:  # Large diff suggesting corrupt data
            warnings.warn('The difference between the mean and median values of the line channel is {}%.\n'
                          .format(diff_between_mean_median * 100))
            line_delta = float(input('If you know the expected time (in seconds) between subsequent lines please write it. Else, write 0:\n'))
            line_delta *= binwidth
            if line_delta <= 0:
                raise ValueError('Line data was corrupt.')
            # Create a new line array
            line_array = create_line_array(last_event_time=last_event_time, num_of_lines=num_of_lines,
                                           num_of_frames=num_of_frames)
            dict_of_data['Lines'] = pd.Series(line_array, name='abs_time', dtype='uint64')
        else:
            # Data is valid. Check whether we need a 0-time line event
            line_delta = median_of_lines
            zeroth_line_delta = np.abs(dict_of_data['Lines'][0] - line_delta)/line_delta
            if zeroth_line_delta < 0.05:
                dict_of_data['Lines'] = pd.Series([0], name='Lines', dtype='uint64') \
                    .append(dict_of_data['Lines'], ignore_index=True)

    else:  # create our own line array
        line_array = create_line_array(last_event_time=last_event_time, num_of_lines=num_of_lines,
                                       num_of_frames=num_of_frames)
        dict_of_data['Lines'] = pd.Series(line_array, name='abs_time', dtype='uint64')
        line_delta = last_event_time/(num_of_lines * int(num_of_frames))

    return dict_of_data, line_delta


def validate_frame_input(dict_of_data: Dict, binwidth, line_delta: int = -1, num_of_lines: int = -1):
    if line_delta == -1:
        raise ValueError('No line delta input received.')

    if num_of_lines == -1:
        raise ValueError('No number of lines received.')

    if 'Frames' in dict_of_data.keys():
        dict_of_data['Frames'] = pd.Series([0], name='abs_time', dtype='uint64')\
            .append(dict_of_data['Frames'], ignore_index=True)
    else:
        last_event_time = int(dict_of_data['Lines'].max() + line_delta)
        frame_array = create_frame_array(lines=dict_of_data['Lines'], last_event_time=last_event_time,
                                         pixels=num_of_lines)
        dict_of_data['Frames'] = pd.Series(frame_array, name='abs_time', dtype='uint64')

    return dict_of_data


def create_frame_array(lines: pd.Series=None, last_event_time: int=None,
                       pixels: int=None) -> np.ndarray:
    """Create a pandas Series of start-of-frame times"""

    if last_event_time is None or pixels is None or lines.empty:
        raise ValueError('Wrong input detected.')

    if last_event_time <= 0:
        raise ValueError('Last event time is zero or negative.')

    num_of_recorded_lines = lines.shape[0]
    actual_num_of_frames = max(num_of_recorded_lines // pixels, 1)

    if num_of_recorded_lines < pixels:
        array_of_frames = np.linspace(start=0, stop=last_event_time, num=int(actual_num_of_frames), endpoint=False)
    else:
        unnecess_lines = num_of_recorded_lines % pixels
        array_of_frames = lines.iloc[0 : int(num_of_recorded_lines-unnecess_lines) : pixels]


    return np.array(array_of_frames)


def create_line_array(last_event_time: int=None, num_of_lines=None, num_of_frames=None) -> np.ndarray:
    """Create a pandas Series of start-of-line times"""

    if (last_event_time is None) or (num_of_lines is None) or (num_of_frames is None):
        raise ValueError('Wrong input detected.')

    if (num_of_lines <= 0) or (num_of_frames <= 0):
        raise ValueError('Number of lines and frames has to be positive.')

    if last_event_time <= 0:
        raise ValueError('Last event time is zero or negative.')

    total_lines = num_of_lines * int(num_of_frames)
    line_array = np.arange(start=0, stop=last_event_time, step=last_event_time/total_lines)
    return line_array


def validate_created_data_channels(dict_of_data: Dict):
    """
    Make sure that the dictionary that contains all data channels makes sense.
    """
    assert {'PMT1', 'Lines', 'Frames'} <= set(dict_of_data.keys())  # A is subset of B

    if dict_of_data['Frames'].shape[0] > dict_of_data['Lines'].shape[0]:  # more frames than lines
        raise UserWarning('More frames than lines, consider replacing the two.')

    try:
        if dict_of_data['TAG Lens'].shape[0] < dict_of_data['Lines'].shape[0]:
            raise UserWarning('More lines than TAG pulses, consider replacing the two.')
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


def validate_laser_input(pulses, laser_freq: float, binwidth: float, offset: int) -> pd.Series:
    """
    Create an orderly laser pulse train.
    :param pulses:
    :param laser_freq:
    :return:
    """
    import warnings

    diffs = pulses.diff()
    pulses_final = pulses[(diffs <= np.ceil((1 / (laser_freq * binwidth)))) &
                          (diffs >= np.floor((1 / (laser_freq * binwidth))))]\
        .reset_index(drop=True) + offset
    if len(pulses_final) < 0.9 * len(pulses):
        warnings.warn("More than 10% of pulses were filtered due to bad timings. Make sure the laser input is fine.")

    pulses_final[0] = pulses[1]
    return pulses_final


def rectify_photons_in_uneven_lines(df: pd.DataFrame, sorted_indices: np.array, lines: pd.Series, bidir: bool = True,
                                    phase: float = 0, keep_unidir: bool = False):
    """
    "Deal" with photons in uneven lines. Unidir - currently throws them away.
    Bidir = flips them over.
    """
    uneven_lines = np.remainder(sorted_indices, 2)
    if bidir:
        time_rel_line = pd.Series(range(df.shape[0]), dtype='int64', name='time_rel_line')
        time_rel_line.loc[uneven_lines == 0] = df.loc[uneven_lines == 0, 'time_rel_line_pre_drop'].values
        # Reverse the relative time of the photons belonging to the uneven lines,
        # by subtracting their relative time from the start time of the next line
        lines_to_subtract_from = lines.loc[sorted_indices[uneven_lines.astype(bool)] + 1].values
        events_to_subtract = df.loc[np.logical_and(uneven_lines, 1), 'abs_time'].values
        time_rel_line.iloc[uneven_lines.nonzero()[0]] = lines_to_subtract_from - events_to_subtract \
            + (np.sin(phase) * lines[1])  # introduce phase delay between lines
        df.insert(loc=len(df.columns), value=time_rel_line.values, column='time_rel_line')

    if not bidir and not keep_unidir:
        df = df.drop(df.index[uneven_lines == 1])
        df['time_rel_line'] = df['time_rel_line_pre_drop']

    if not bidir and keep_unidir:  # Unify the excess rows and photons in them into the previous row
        sorted_indices[np.logical_and(uneven_lines, 1)] -= 1
        df['Lines'] = lines.loc[sorted_indices].values

    df.drop(['time_rel_line_pre_drop'], axis=1, inplace=True)
    df = df[df.loc[:, 'time_rel_line'] >= 0]

    return df
