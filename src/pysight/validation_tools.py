"""
__author__ = Hagai Hargil
"""
from typing import Dict, List
import pandas as pd
import numpy as np
import warnings
from enum import Enum, auto


class LineDataType(Enum):
    CORRUPT = auto()
    REBUILD = auto()
    VALID = auto()
    FROM_SCRATCH = auto()
    SWEEPS_REBUILD = auto()
    SWEEPS_FROM_SCRATCH = auto()
    ADD = auto()


def validate_line_input(dict_of_data: Dict, cols_in_data: List, num_of_lines: int=-1,
                        num_of_frames: int=-1, binwidth: float=800e-12,
                        last_event_time: int=-1, line_freq: float=7930.0, bidir: bool=False,
                        delay_between_frames: float=0.0011355, use_sweeps: bool=False,
                        max_sweep: int=65535, total_sweep_time: int=1, bidir_phase: float=-2.79):
    """
    Verify that the .lst input of lines exists and looks fine. Create one if there's no such input.
    """
    from pysight.line_process import LineProcess

    if num_of_lines == -1:
        raise ValueError('No number of lines input received.')

    if num_of_frames == -1:
        raise ValueError('No number of frames received.')

    if last_event_time == -1:
        raise ValueError('No last event time received.')

    if len(cols_in_data) == 0:
        raise ValueError('No columns in data.')

    # Find the suitable case for this data regarding the line signal recorded
    type_of_line_data = match_line_data_to_case(lines=dict_of_data.get('Lines'),
                                                num_of_lines=num_of_lines,
                                                keys=list(dict_of_data.keys()),
                                                use_sweeps=use_sweeps, bidir=bidir,
                                                bidir_phase=bidir_phase, binwidth=binwidth)

    dict_of_data, line_delta = LineProcess(line_freq=line_freq, binwidth=binwidth,
                                           bidir=bidir, last_event_time=last_event_time,
                                           dict_of_data=dict_of_data,
                                           num_of_lines=num_of_lines,
                                           delay_between_frames=delay_between_frames,
                                           num_of_frames=num_of_frames,
                                           cols_in_data=cols_in_data, max_sweep=max_sweep,
                                           total_sweep_time=total_sweep_time,
                                           bidir_phase=bidir_phase)\
        .process(type_of_line_data)
    return dict_of_data, line_delta


def validate_frame_input(dict_of_data: Dict, binwidth, cols_in_data: List, num_of_lines: int=-1,
                         last_event_time: int=-1):

    if num_of_lines == -1:
        raise ValueError('No number of lines received.')

    if last_event_time == -1:
        raise ValueError('No last event time input received.')

    if len(cols_in_data) == 0:
        raise ValueError('No columns in data.')

    if 'Frames' in dict_of_data.keys():
        dict_of_data['Frames'] = pd.DataFrame([[0] * len(cols_in_data)],
                                                     columns=cols_in_data,
                                                     dtype='uint64')\
                    .append(dict_of_data['Frames'], ignore_index=True)
    else:
        frame_array = create_frame_array(lines=dict_of_data['Lines'].loc[:, 'abs_time'],
                                         last_event_time=last_event_time,
                                         pixels=num_of_lines)
        dict_of_data['Frames'] = pd.DataFrame(frame_array, columns=['abs_time'], dtype='uint64')

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
        array_of_frames = np.linspace(start=0, stop=last_event_time, num=int(actual_num_of_frames),
                                      endpoint=False, dtype=np.uint64)
    else:
        unnecess_lines = num_of_recorded_lines % pixels
        array_of_frames = lines.iloc[0 : int(num_of_recorded_lines-unnecess_lines) : pixels]

    return np.array(array_of_frames)

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


def validate_laser_input(pulses, laser_freq: float, binwidth: float) -> pd.Series:
    """
    Create an orderly laser pulse train.
    :param pulses:
    :param laser_freq:
    :return:
    """
    import warnings

    diffs = pulses.loc[:, 'abs_time'].diff()
    rel_idx = (diffs <= 1.05*np.ceil((1 / (laser_freq * binwidth)))) & (diffs >= 0.95*np.floor((1 / (laser_freq * binwidth))))
    pulses_final = pulses[rel_idx]  # REMINDER: Laser offset wasn't added
    if len(pulses_final) < 0.9 * len(pulses):  # TODO: If there's holdafter time, there's a good chance that the pulses
                                               # will not be periodic due to this extra time.
        warnings.warn("More than 10% of pulses were filtered due to bad timings. Make sure the laser input is fine.")

    pulses_final = pd.concat([pulses.loc[:0, :], pulses_final])  # Add back the first pulse
    pulses_final.reset_index(drop=True, inplace=True)

    return pulses_final


def rectify_photons_in_uneven_lines(df: pd.DataFrame, sorted_indices: np.array, lines: pd.Series, bidir: bool=True,
                                    keep_unidir: bool=False):
    """
    "Deal" with photons in uneven lines. Unidir - if keep_unidir is false, will throw them away.
    Bidir = flips them over (in the Volume object)
    """
    uneven_lines = np.remainder(sorted_indices, 2)
    if bidir:
        df.rename(columns={'time_rel_line_pre_drop': 'time_rel_line'}, inplace=True)

    elif not bidir and not keep_unidir:
        df = df.iloc[uneven_lines != 1, :].copy()
        df.rename(columns={'time_rel_line_pre_drop': 'time_rel_line'}, inplace=True)

    elif not bidir and keep_unidir:  # Unify the excess rows and photons in them into the previous row
        sorted_indices[np.logical_and(uneven_lines, 1)] -= 1
        df.loc['Lines'] = lines.loc[sorted_indices].values

    try:
        df.drop(['time_rel_line_pre_drop'], axis=1, inplace=True)
    except ValueError:  # column label doesn't exist
        pass
    df = df.loc[df.loc[:, 'time_rel_line'] >= 0, :]

    return df


def calc_last_event_time(dict_of_data: Dict, lines_per_frame: int=-1):
    """
    Find the last event time for the experiment. Logic as follows:
    No lines \ frames data given: Last event time is the last photon time.
    Only lines data given: The last start-of-frame time is created, and the difference between subsequent frames
    in the data is added.
    Frames data exists: The last frame time plus the difference between subsequent frames is the last event time.
    :param dict_of_data: Dictionary of data.
    :param lines_per_frame: Lines per frame.
    :return: int
    """

    # Basic assertions
    if lines_per_frame < 1:
        raise ValueError('No lines per frame value received, or value was corrupt.')

    if 'PMT1' not in dict_of_data:
        raise ValueError('No PMT1 channel in dict_of_data.')

    ##
    if 'Frames' in dict_of_data:
        last_frame_time = dict_of_data['Frames'].loc[:, 'abs_time'].iloc[-1]

        if dict_of_data['Frames'].shape[0] == 1:
            return int(2 * last_frame_time)
        else:
            frame_diff = int(dict_of_data['Frames'].loc[:, 'abs_time'].diff().mean())
            return int(last_frame_time + frame_diff)

    if 'Lines' in dict_of_data:
        num_of_lines_recorded = dict_of_data['Lines'].shape[0]
        div, mod = divmod(num_of_lines_recorded, lines_per_frame)

        if num_of_lines_recorded > lines_per_frame * (div+1):  # excessive number of lines
            last_line_of_last_frame = dict_of_data['Lines'].loc[:, 'abs_time']\
                .iloc[div * lines_per_frame - 1]
            frame_diff = dict_of_data['Lines'].loc[:, 'abs_time'].iloc[div * lines_per_frame - 1] -\
                dict_of_data['Lines'].loc[:, 'abs_time'].iloc[(div - 1) * lines_per_frame]
            return int(last_line_of_last_frame + frame_diff)

        elif mod == 0:  # number of lines contained exactly in number of lines per frame
            return int(dict_of_data['Lines'].loc[:, 'abs_time'].iloc[-1] + dict_of_data['Lines']\
                .loc[:, 'abs_time'].diff().mean())

        elif num_of_lines_recorded < lines_per_frame * (div+1):
            missing_lines = lines_per_frame - mod
            line_diff = int(dict_of_data['Lines'].loc[:, 'abs_time'].diff().mean())
            return int(dict_of_data['Lines'].loc[:, 'abs_time'].iloc[-1] +\
                ((missing_lines+1) * line_diff))

    # Just PMT data
    max_pmt1 = dict_of_data['PMT1'].loc[:, 'abs_time'].max()
    try:
        max_pmt2 = dict_of_data['PMT2'].loc[:, 'abs_time'].max()
    except KeyError:
        return max_pmt1
    else:
        return max(max_pmt1, max_pmt2)


def match_line_data_to_case(lines: pd.Series, keys: list, num_of_lines: int=512,
                            use_sweeps: bool=False, bidir: bool=False, binwidth: float=800e-12,
                            bidir_phase: float=-2.79) -> LineDataType:
    """
    Enumerate all possibilities of line data and choose the right option
    :param lines: Line data
    :param keys: Keys of `dict_of_data` dictionary
    :param num_of_lines: Number of lines per frame
    :param use_sweeps: Whether the sweeps have a meaning for image generation.
    :return: String of the specific case. Either: 'corrupt', 'rebuild', 'valid', 'from_scratch',
    'sweeps-rebuild' or 'sweeps-from-scratch'.
    """

    if use_sweeps and 'Lines' in keys:
        return LineDataType.SWEEPS_REBUILD

    if use_sweeps and not 'Lines' in keys:
        return LineDataType.SWEEPS_FROM_SCRATCH

    if 'Lines' in keys:
        return find_line_case(lines=lines.loc[:, 'abs_time'].copy(),
                              num_of_lines=num_of_lines, keys=keys)
    else:
        return LineDataType.FROM_SCRATCH


def add_phase_to_bidir_lines(lines: pd.Series, bidir_phase: float=-2.79, binwidth: float=800e-12):
    """
    "Fix" temporarily the lines for them to pass the corruption check
    :param lines: Lines after the phase change
    :param bidir_phase:
    :param binwidth:
    :return: Lines "pre-fix" of phase change - their original form
    """
    phase_in_seconds = bidir_phase * 1e-6
    if phase_in_seconds < 0:
        lines.iloc[1::2] -= np.uint64(np.abs(phase_in_seconds / binwidth))
    else:
        lines.iloc[1::2] += np.uint64(phase_in_seconds / binwidth)
    return lines


def find_line_case(lines: pd.Series, num_of_lines: int,
                   keys:list,) -> LineDataType:
    """
    Match the right way to process lines when a line signal was recorded.
    :param lines: Line data
    :param num_of_lines: The number of lines per frame
    :param keys: Input data signals
    :return: LineDataType
    """
    # Verify that the input is not corrupt
    if lines.shape[0] < num_of_lines // 2:
        warnings.warn("Line data was corrupt as there were too few lines.\n"
                      "Simulating line data using GUI's parameters.")
        return LineDataType.CORRUPT

    change_thresh = 0.5
    max_change_pct = lines[lines.diff().pct_change(periods=1) > change_thresh]
    mean_max_change = np.diff(max_change_pct.diff().index.values).mean()

    if len(max_change_pct) / lines.shape[0] > change_thresh and 'Frames' not in keys:  # 0.1
        warnings.warn("Line data was corrupt - the period didn't make sense.\n"
                      f"{len(max_change_pct)} out of {lines.shape[0]} lines were mispositioned. "
                      "Simulating line data using GUI's parameters.")
        return LineDataType.CORRUPT

    elif len(max_change_pct) / lines.shape[0] > change_thresh and 'Frames' in keys:
        return LineDataType.REBUILD

    elif 0.95 < mean_max_change / num_of_lines < 1.05 and max_change_pct.diff().index[0] != num_of_lines:
        # Lines were missing from the start of the recording, but otherwise the data is clean
        return LineDataType.ADD

    elif len(max_change_pct) / lines.shape[0] < change_thresh:
        return LineDataType.VALID
