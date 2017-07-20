"""
__author__ = Hagai Hargil
"""
from typing import Dict, List
import pandas as pd
import numpy as np
import warnings


def validate_line_input(dict_of_data: Dict, cols_in_data: List, num_of_lines: int=-1,
                        num_of_frames: int=-1, binwidth: float=800e-12,
                        last_event_time: int=-1, line_freq: float=7930.0, bidir: bool=False,
                        delay_between_frames: float=0.0011355, use_sweeps: bool=False,
                        max_sweep: int=65535, total_sweep_time: int=1):
    """ Verify that the .lst input of lines exists and looks fine. Create one if there's no such input. """
    if num_of_lines == -1:
        raise ValueError('No number of lines input received.')

    if num_of_frames == -1:
        raise ValueError('No number of frames received.')

    if last_event_time == -1:
        raise ValueError('No last event time received.')

    if len(cols_in_data) == 0:
        raise ValueError('No columns in data.')

    # Find the suitable case for this data regarding the line signal recorded
    type_of_line_data = match_line_data_to_case(lines=dict_of_data['Lines'].loc[:, 'abs_time'],
                                                num_of_lines=num_of_lines,
                                                keys=list(dict_of_data.keys()),
                                                use_sweeps=use_sweeps)

    if 'corrupt' == type_of_line_data:
        # Data is corrupted, and no frame channel can help us.
        line_delta = bins_bet_lines(line_freq=line_freq, binwidth=binwidth,
                                    bidir=bidir)
        dict_of_data['Lines'] = extrapolate_line_data(last_event=last_event_time,
                                                      line_point=dict_of_data['Lines'].at[0, 'abs_time'],
                                                      num_of_lines=num_of_lines,
                                                      line_delta=line_delta,
                                                      delay_between_frames=delay_between_frames,
                                                      bidir=bidir, binwidth=binwidth,
                                                      num_of_frames=num_of_frames)
        return dict_of_data, line_delta


    elif 'rebuild' == type_of_line_data:
        # Data is corrupted, but we can rebuild lines on top of the frame channel
        line_array = create_line_array(last_event_time=last_event_time, num_of_lines=num_of_lines,
                                       num_of_frames=num_of_frames)
        dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        line_delta = last_event_time / (num_of_lines * int(num_of_frames))
        return dict_of_data, line_delta

    elif 'valid' == type_of_line_data:
        # Data is valid. Check whether we need a 0-time line event
        line_delta = dict_of_data['Lines'].loc[:, 'abs_time'].diff().median()
        zeroth_line_delta = np.abs(dict_of_data['Lines'].loc[0, 'abs_time'] - line_delta)/line_delta
        if zeroth_line_delta < 0.05:
            dict_of_data['Lines'] = pd.DataFrame([[0] * len(cols_in_data)],
                                                 columns=cols_in_data,
                                                 dtype='uint64')\
                .append(dict_of_data['Lines'], ignore_index=True)
        return dict_of_data, line_delta

    elif 'from_scratch' == type_of_line_data:  # create our own line array
        line_array = create_line_array(last_event_time=last_event_time, num_of_lines=num_of_lines,
                                       num_of_frames=num_of_frames)
        dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        line_delta = last_event_time/(num_of_lines * int(num_of_frames))
        return dict_of_data, line_delta

    elif 'sweeps-rebuild' == type_of_line_data:  # create our own line array with the sweeps data
        line_delta = 1 / line_freq if not bidir else 1 / (2 * line_freq)
        num_of_frames = np.ceil(last_event_time / (line_delta / binwidth * num_of_lines))
        line_array = create_line_array(last_event_time=last_event_time, num_of_lines=num_of_lines,
                                       num_of_frames=num_of_frames, line_delta=line_delta/binwidth)
        dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        return dict_of_data, int(line_delta / binwidth)

    elif 'sweeps-from-scratch' == type_of_line_data:  # brand new line data
        sweep_vec = np.arange(max_sweep + 1, dtype=np.uint64)
        if len(sweep_vec) < 2:
            warnings.warn("All data was registered to a single sweep. Line data will be completely simulated.")
        else:
            dict_of_data['Lines'] = pd.DataFrame(
                sweep_vec * total_sweep_time,
                columns=['abs_time'], dtype=np.uint64)
        return dict_of_data, total_sweep_time

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


def bins_bet_lines(line_freq: float=0, binwidth: float=0,
                   bidir: bool=False) -> int:
    """
    Calculate the line difference in bins, according to the frequency
    of the line signal.
    :param line_freq: Frequency of lines in Hz. Regardless of bidirectional scanning.
    :return: int - line delta
    """
    freq_in_bins = 1/(line_freq * binwidth)
    return int(freq_in_bins / 2) if bidir else freq_in_bins


def extrapolate_line_data(last_event: int, line_point: int=0,
                          line_delta: int=0, num_of_lines: int=1,
                          delay_between_frames: float=0.0011355,
                          bidir: bool=False, binwidth: float=800e-12,
                          num_of_frames: int=1) -> pd.DataFrame:
    """
    From a single line signal extrapolate the presumed line data vector. The line frequency is doubled
    the original frequency. If needed, events will be discarded later.
    :param last_event: The last moment of the experiment
    :param line_delta: Bins between subsequent lines.
    :param line_point: Start interpolation from this point.
    :param num_of_lines: Number of lines in a frame.
    :param delay_between_frames: Time (in sec) between frames.
    :param bidir: Whether the scan was bidirectional.
    :param binwidth: Binwidth of multiscaler in seconds.
    :return: pd.DataFrame of line data
    """
    # Verify input
    if line_point / line_delta > num_of_lines:  # The first recorded line came very late
        line_point = 0

    # Create the matrix containing the duplicate frame data
    delay_between_frames_in_bins = int(delay_between_frames / binwidth)
    time_of_frame = line_delta * num_of_lines \
                    + delay_between_frames_in_bins
    num_of_frames = max(int(np.floor(last_event / time_of_frame)), 1)
    time_of_frame_mat = np.tile(time_of_frame * np.arange(num_of_frames), (num_of_lines, 1))

    # Create the matrix containing the duplicate line data
    line_vec = np.arange(start=line_point, stop=line_delta*num_of_lines, step=line_delta,
                         dtype=np.uint64)
    line_vec = np.r_[np.flip(np.arange(start=line_point, stop=0, step=-line_delta,
                               dtype=np.uint64)[1:], axis=0), line_vec]
    # Check if 0 should be included
    if line_vec[0] - line_delta == 0:
        line_vec = np.r_[0, line_vec]

    line_mat = np.tile(line_vec.reshape(len(line_vec), 1), (1, num_of_frames))

    # Add them up
    assert len(line_mat) > 0
    assert len(time_of_frame_mat) > 0
    final_mat = line_mat + time_of_frame_mat
    line_vec_final = np.ravel(final_mat, order='F')

    return pd.DataFrame(line_vec_final, columns=['abs_time'], dtype=np.uint64)


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


def create_line_array(last_event_time: int=None, num_of_lines=None, num_of_frames=None,
                      line_delta: float=0.0) -> np.ndarray:
    """Create a pandas Series of start-of-line times"""

    if (last_event_time is None) or (num_of_lines is None) or (num_of_frames is None):
        raise ValueError('Wrong input detected.')

    if (num_of_lines <= 0) or (num_of_frames <= 0):
        raise ValueError('Number of lines and frames has to be positive.')

    if last_event_time <= 0:
        raise ValueError('Last event time is zero or negative.')

    total_lines = num_of_lines * int(num_of_frames)
    if line_delta == 0.0:
        line_array = np.arange(start=0, stop=last_event_time, step=last_event_time/total_lines, dtype=np.uint64)
    else:
        line_array = np.arange(start=0, stop=last_event_time, step=line_delta, dtype=np.uint64)
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

    diffs = pulses.loc[:, 'abs_time'].diff()
    rel_idx = (diffs <= np.ceil((1 / (laser_freq * binwidth)))) & (diffs >= np.floor((1 / (laser_freq * binwidth))))
    pulses_final = pulses[rel_idx]  # REMINDER: Laser offset wasn't added
    if len(pulses_final) < 0.9 * len(pulses):
        warnings.warn("More than 10% of pulses were filtered due to bad timings. Make sure the laser input is fine.")

    pulses_final = pd.concat([pulses.loc[:0, :], pulses_final])  # Add back the first pulse
    pulses_final.reset_index(drop=True, inplace=True)

    return pulses_final


def rectify_photons_in_uneven_lines(df: pd.DataFrame, sorted_indices: np.array, lines: pd.Series, bidir: bool = True,
                                    phase: float = 0, keep_unidir: bool = False):
    """
    "Deal" with photons in uneven lines. Unidir - if keep_unidir is false, will throw them away.
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
        df = df.loc[uneven_lines != 1, :].copy()
        df.rename(columns={'time_rel_line_pre_drop': 'time_rel_line'}, inplace=True)

    if not bidir and keep_unidir:  # Unify the excess rows and photons in them into the previous row
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
                            use_sweeps:bool=False) -> str:
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
        return 'sweeps-rebuild'

    if use_sweeps and not 'Lines' in keys:
        return 'sweeps-from-scratch'

    if 'Lines' in keys:
        # Verify that the input is not corrupt
        if lines.shape[0] < num_of_lines // 2:
            warnings.warn("Line data was corrupt as there were too few lines.\n"
                          "Simulating line data using GUI's parameters.")
            return 'corrupt'

        max_change_pct = lines[lines.diff().pct_change(periods=1) > 0.05]
        if len(max_change_pct) / lines.shape[0] > 0.1 and 'Frames' not in keys:
            warnings.warn("Line data was corrupt - the period didn't make sense.\n"
                          f"{len(max_change_pct)} out of {lines.shape[0]} lines were mispositioned. "
                          "Simulating line data using GUI's parameters.")
            return 'corrupt'

        elif len(max_change_pct) / lines.shape[0] > 0.1 and 'Frames' in keys:
            return 'rebuild'

        elif len(max_change_pct) / lines.shape[0] < 0.1:
            return 'valid'

    else:
        return 'from_scratch'
