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
            dict_of_data['Lines'] = pd.Series(line_array, name='abs_time', dtype=np.uint64)
        else:
            line_delta = median_of_lines

    else:  # create our own line array
        line_array = create_line_array(last_event_time=last_event_time, num_of_lines=num_of_lines,
                                       num_of_frames=num_of_frames)
        dict_of_data['Lines'] = pd.Series(line_array, name='abs_time', dtype=np.uint64)
        line_delta = dict_of_data['Lines'].diff().mean()

    return dict_of_data, line_delta


def validate_frame_input(dict_of_data: Dict, flyback: float = 0.001, binwidth: float = 800e-12,
                         line_delta: int = -1, num_of_lines: int = -1):
    if line_delta == -1:
        raise ValueError('No line delta input received.')

    if num_of_lines == -1:
        raise ValueError('No number of lines received.')

    if 'Frames' in dict_of_data.keys():
        dict_of_data['Frames'] = pd.Series([0], name='abs_time', dtype=np.uint64)\
            .append(dict_of_data['Frames'], ignore_index=True)
    else:
        last_event_time = int(dict_of_data['Lines'].max() + line_delta)
        frame_array = create_frame_array(lines=dict_of_data['Lines'], last_event_time=last_event_time,
                                         pixels=num_of_lines, spacing_between_lines=line_delta,
                                         flyback=flyback, binwidth=binwidth)
        dict_of_data['Frames'] = pd.Series(frame_array, name='abs_time', dtype=np.uint64)

    return dict_of_data


def create_frame_array(lines: pd.Series=None, last_event_time: int=None,
                       pixels: int=None, spacing_between_lines: float=None,
                       flyback: float=0.001, binwidth: float=800e-12) -> np.ndarray:
    """Create a pandas Series of start-of-frame times"""

    if last_event_time is None or pixels is None or lines.empty:
        raise ValueError('Wrong input detected.')

    if last_event_time <= 0:
        raise ValueError('Last event time is zero or negative.')

    num_of_recorded_lines = lines.shape[0]
    actual_num_of_frames = max(num_of_recorded_lines // pixels, 1)
    if num_of_recorded_lines < pixels:
        unnecess_lines = 0
    else:
        unnecess_lines = actual_num_of_frames % pixels

    if unnecess_lines == 0:  # either less lines than pixels or a integer multiple of pixels
        array_of_frames = np.linspace(start=0, stop=last_event_time, num=int(actual_num_of_frames), endpoint=False)
    else:
        last_event_time = int(lines.iloc[num_of_recorded_lines - unnecess_lines] + spacing_between_lines)
        array_of_frames = np.linspace(start=0, stop=last_event_time, num=int(actual_num_of_frames), endpoint=False)

    # Add flyback consideration
    # for idx, frame_start in enumerate(array_of_frames[1:], 1):
    #     array_of_frames[idx] = frame_start + ((flyback * idx) / binwidth)

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


def validate_laser_input(pulses, laser_freq: float, binwidth: float) -> pd.Series:
    """
    Create an orderly laser pulse train.
    :param pulses:
    :param laser_freq:
    :return:
    """
    import warnings


    diffs = pulses.diff()
    pulses_final = pulses[(diffs < np.ceil((1 / (laser_freq * binwidth)))) & (diffs >= 0)].reset_index(drop=True)
    if len(pulses_final) < 0.9 * len(pulses):
        warnings.warn("More than 10% of pulses were filtered due to bad timings. Make sure the laser input is fine.")

    return pulses_final
