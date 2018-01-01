"""
__author__ = Hagai Hargil
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import warnings
import attr
from attr.validators import instance_of
from enum import Enum
from pysight.tkinter_gui_multiscaler import ImagingSoftware
from itertools import tee, chain


class LinesType(Enum):
    SWEEPS_REBUILD = 'sweeps-rebuild'
    SWEEPS_FROM_SCRATCH = 'sweeps-from-scratch'
    CORRUPT = 'corrupt'
    REBUILD = 'rebuild'
    SCANIMAGE = 'si_valid'
    MSCAN = 'mscan_valid'
    FROM_SCRATCH = 'from_scratch'


@attr.s(slots=True)
class SignalValidator:
    """
    Parse a dictionary of all recorded signals and verify the data's integrity
    """
    dict_of_data = attr.ib(validator=instance_of(dict))
    data_to_grab = attr.ib(default=['abs_time', 'sweep'], validator=instance_of(list))
    num_of_lines = attr.ib(default=512, validator=instance_of(int))
    num_of_frames = attr.ib(default=1, validator=instance_of(int))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    line_freq = attr.ib(default=7900.0, validator=instance_of(float))
    bidir = attr.ib(default=False, validator=instance_of(bool))
    delay_between_frames = attr.ib(default=0.0011355, validator=instance_of(float))
    use_sweeps = attr.ib(default=False, validator=instance_of(bool))
    bidir_phase = attr.ib(default=-2.7, validator=instance_of(float))
    image_soft = attr.ib(default=ImagingSoftware.SCANIMAGE.value, validator=instance_of(str))
    handle_line_cases = attr.ib(init=False)
    last_event_time = attr.ib(init=False)
    max_sweep = attr.ib(init=False)
    line_type_data = attr.ib(init=False)
    line_delta = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.handle_line_cases = {
            LinesType.CORRUPT: self.__corrupt,
            LinesType.FROM_SCRATCH: self.__from_scratch,
            LinesType.SWEEPS_FROM_SCRATCH: self.__sweeps_from_scratch,
            LinesType.REBUILD: self.__rebuild,
            LinesType.SWEEPS_REBUILD: self.__sweeps_rebuild,
            LinesType.SCANIMAGE: self.__si_wrapper,
            LinesType.MSCAN: self.__mscan_wrapper,

        }

    @property
    def frame_delay(self):
        return np.uint64(self.delay_between_frames / self.binwidth)

    @property
    def total_sweep_time(self):
        return self.acq_delay + self.data_range + self.time_after_sweep

    @property
    def max_sweep(self):
        return self.df.sweep.max()

    def run(self):
        """
        Main pipeline
        :return:
        """
        if 'Frames' in self.dict_of_data:
            self.num_of_frames = self.dict_of_data['Frames'].shape[0] + 1  # account for first frame

        self.last_event_time = self.__calc_last_event_time()

        self.line_type_data: LinesType = \
            self.__match_line_data_to_case(lines=self.dict_of_data.get('Lines'))

        self.dict_of_data, self.line_delta = \
            self.handle_line_cases[self.line_type_data]()

        self.dict_of_data = self.__validate_frame_input()

        try:
            self.dict_of_data['Laser'] = self.__validate_laser_input(self.dict_of_data['Laser'])
        except KeyError:
            pass

        self.__validate_created_data_channels()

    @staticmethod
    def __pairwise(iterable):
        """From itertools: s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def __bins_bet_lines(self) -> int:
        """
        Calculate the line difference in bins, according to the frequency
        of the line signal.
        :param line_freq: Frequency of lines in Hz. Regardless of bidirectional scanning.
        :return: int - line delta
        """
        freq_in_bins = 1 / (self.line_freq * self.binwidth)
        return int(freq_in_bins / 2) if self.bidir else freq_in_bins

    def __calc_last_event_time(self):
        """
        Find the last event time for the experiment. Logic as follows:
        No lines \ frames data given: Last event time is the last photon time.
        Only lines data given: The last start-of-frame time is created, and the difference between subsequent frames
        in the data is added.
        Frames data exists: The last frame time plus the difference between subsequent frames is the last event time.
        :param dict_of_data: Dictionary of data.
        :param self.num_of_lines: Lines per frame.
        :return: int
        """
        ##
        if 'Frames' in self.dict_of_data:
            last_frame_time = self.dict_of_data['Frames'].loc[:, 'abs_time'].iloc[-1]

            if self.dict_of_data['Frames'].shape[0] == 1:
                return int(2 * last_frame_time)
            else:
                frame_diff = int(self.dict_of_data['Frames'].loc[:, 'abs_time'].diff().mean())
                return int(last_frame_time + frame_diff)

        if 'Lines' in self.dict_of_data:
            num_of_lines_recorded = self.dict_of_data['Lines'].shape[0]
            div, mod = divmod(num_of_lines_recorded, self.num_of_lines)

            if num_of_lines_recorded > self.num_of_lines * (div+1):  # excessive number of lines
                last_line_of_last_frame = self.dict_of_data['Lines'].loc[:, 'abs_time']\
                    .iloc[div * self.num_of_lines - 1]
                frame_diff = self.dict_of_data['Lines'].loc[:, 'abs_time'].iloc[div * self.num_of_lines - 1] -\
                    self.dict_of_data['Lines'].loc[:, 'abs_time'].iloc[(div - 1) * self.num_of_lines]
                return int(last_line_of_last_frame + frame_diff)

            elif mod == 0:  # number of lines contained exactly in number of lines per frame
                return int(self.dict_of_data['Lines'].loc[:, 'abs_time'].iloc[-1] + self.dict_of_data['Lines']\
                    .loc[:, 'abs_time'].diff().mean())

            elif num_of_lines_recorded < self.num_of_lines * (div+1):
                missing_lines = self.num_of_lines - mod
                line_diff = int(self.dict_of_data['Lines'].loc[:, 'abs_time'].diff().mean())
                return int(self.dict_of_data['Lines'].loc[:, 'abs_time'].iloc[-1] +\
                    ((missing_lines+1) * line_diff))

        # Just PMT data
        max_pmt1 = self.dict_of_data['PMT1'].loc[:, 'abs_time'].max()
        try:
            max_pmt2 = self.dict_of_data['PMT2'].loc[:, 'abs_time'].max()
        except KeyError:
            return max_pmt1
        else:
            return max(max_pmt1, max_pmt2)

    def __match_line_data_to_case(self, lines: pd.Series) -> LinesType:
        """
        Find the suitable case for this data regarding the line signal recorded
        :param lines: Line signal
        :return: LinesType enumeration
        """
        if self.use_sweeps and 'Lines' in self.dict_of_data.keys():
            return LinesType.SWEEPS_REBUILD

        if self.use_sweeps and not 'Lines' in self.dict_of_data.keys():
            return LinesType.SWEEPS_FROM_SCRATCH

        if 'Lines' in self.dict_of_data.keys():
            lines = lines.loc[:, 'abs_time'].copy()
            # Verify that the input is not corrupt
            if lines.shape[0] < self.num_of_lines // 2:
                warnings.warn("Line data was corrupt - there were too few lines.\n"
                              "Simulating line data using GUI's parameters.")

                if 'Frames' in self.dict_of_data.keys():
                    return LinesType.REBUILD
                else:
                    return LinesType.CORRUPT

            lines = self.__add_zeroth_line_event(lines=lines)

            # Analyze the line signal
            change_thresh = 0.3
            rel_idx = np.where(np.abs(lines.diff().pct_change(periods=1)) > change_thresh)[0]
            is_corrupt_sig: bool = len(rel_idx) / lines.shape[0] > change_thresh
            if is_corrupt_sig and 'Frames' in self.dict_of_data.keys():
                return LinesType.REBUILD
            elif is_corrupt_sig and 'Frames' not in self.dict_of_data.keys():
                return LinesType.CORRUPT
            else:  # not corrupt
                if self.image_soft == ImagingSoftware.SCANIMAGE.value:
                    return LinesType.SCANIMAGE
                elif self.image_soft == ImagingSoftware.MSCAN.value:
                    return LinesType.MSCAN

        else:
            return LinesType.FROM_SCRATCH

    def __si_wrapper(self) -> Tuple[Dict[str, pd.DataFrame], float]:
        """
        Interpolate SI=specific line signals
        :return:
        """
        lines = self.dict_of_data['Lines'].loc[:, 'abs_time'].copy()
        lines_mat, rel_idx, end_of_frames_idx, last_idx_of_row, rel_idx_non_end_frame = \
            self.__calc_line_parameters_si(lines=lines)
        theo_lines, delta = self.__gen_line_model_si(lines=lines, rel_idx=rel_idx, end_of_frames_idx=end_of_frames_idx)
        lines_mat = self.__diff_mat_analysis_si(y=theo_lines, lines_mat=lines_mat, last_idx=last_idx_of_row,
                                                delta=delta, rel_idx=rel_idx_non_end_frame)
        lines = self.__finalize_lines_si(lines_mat=lines_mat)
        if self.bidir:
            lines = self.__add_phase_to_bidir_lines(lines=lines)
        self.dict_of_data['Lines'] = pd.DataFrame(lines, dtype=np.uint64, columns=['abs_time'])
        return self.dict_of_data, delta

    def __mscan_wrapper(self) -> Tuple[Dict[str, pd.DataFrame], float]:
        """
        Interpolate MScan-specific line signals
        :return:
        """
        lines = self.dict_of_data['Lines'].loc[:, 'abs_time'].copy()
        rel_idx, delta = self.__calc_line_parameters_mscan(lines=lines)
        if len(rel_idx) > 0:
            theo_lines = self.__gen_line_model_mscan(lines=lines, m=delta)
            lines = self.__diff_vec_analysis_mscan(lines=lines, y=theo_lines, delta=delta)
        lines = self.__finalize_lines_mscan(lines=lines, delta=delta)
        if self.bidir:
            lines = self.__add_phase_to_bidir_lines(lines=lines)

        self.dict_of_data['Lines'] = pd.DataFrame(lines, dtype=np.uint64, columns=['abs_time'])
        return self.dict_of_data, delta

    def __calc_line_parameters_si(self, lines: pd.Series) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray, np.ndarray]:
        """ Generate general parameters of the given acquisition """
        change_thresh = 0.3
        rel_idx = np.where(np.abs(lines.diff().pct_change(periods=1)) > change_thresh)[0]
        end_of_frames = rel_idx > 5
        end_of_frames_idx = rel_idx[end_of_frames][::2]  # scanimage specific
        rel_idx_non_end_frame = rel_idx[np.logical_not(end_of_frames)]
        time_between_frames = np.uint64(lines.diff()[end_of_frames_idx].median())
        idx_list = [slice(st, sp) for st, sp in self.__pairwise([0] + list(end_of_frames_idx))]
        lines_mat = np.zeros((len(idx_list), self.num_of_lines * 2), dtype=np.uint64)
        last_idx_of_row = np.zeros((end_of_frames_idx.shape[0]), dtype=np.int32)
        for cur_row, cur_slice in enumerate(idx_list):
            last_idx = cur_slice.stop-cur_slice.start
            lines_mat[cur_row, :last_idx] = lines.values[cur_slice]
            last_idx_of_row[cur_row] = last_idx
        start_time_of_frames = lines[end_of_frames_idx].values
        start_time_of_frames = np.concatenate((np.array([0], dtype=np.uint64), start_time_of_frames[:-1])) \
            .reshape((end_of_frames_idx.shape[0], 1))
        lines_mat -= start_time_of_frames
        if 'Frames' not in self.dict_of_data:
            self.dict_of_data['Frames'] = pd.DataFrame(start_time_of_frames.ravel(), columns=['abs_time'])

        for row, last_idx in enumerate(last_idx_of_row):
            lines_mat[row, last_idx:] = 0

        return lines_mat, rel_idx, end_of_frames_idx, last_idx_of_row, rel_idx_non_end_frame

    def __calc_line_parameters_mscan(self, lines: pd.Series) -> Tuple[np.ndarray, np.uint64]:
        """ Generate general parameters of the given acquisition """
        change_thresh = 0.3
        rel_idx = np.where(np.abs(lines.diff().pct_change(periods=1)) > change_thresh)[0]
        delta = np.uint64(lines.drop(rel_idx).diff().median())
        return rel_idx[::2], delta

    def __gen_line_model_mscan(self, lines: pd.Series, m: np.uint64) -> np.ndarray:
        """ Using linear approximation generate a model for the "correct" line signal """
        const = lines.iloc[0]
        x = np.arange(start=0, stop=len(lines), dtype=np.uint64)
        y = m * x + const

        # MScan's lines are evenly separated
        first_diff = np.uint64(lines.iloc[10:110].diff()[1::2].median())
        delta_diff = np.int32((m - first_diff) / 2)
        if first_diff < m:
            y[1::2] -= delta_diff
            y[::2] += delta_diff
        else:
            y[1::2] += delta_diff
            y[::2] -= delta_diff
        return y

    def __gen_line_model_si(self, lines: pd.Series, rel_idx: np.ndarray, end_of_frames_idx: np.ndarray):
        """ Using linear approximation generate a model for the "correct" line signal """
        rel_idx_first_frame = rel_idx[rel_idx < self.num_of_lines]
        lines_first_frame = lines.iloc[:self.num_of_lines].drop(rel_idx_first_frame)
        # Create a model: y = mx + const
        const = lines_first_frame[0]
        m = np.uint64(lines_first_frame.diff().median())
        x = np.arange(start=0, stop=self.num_of_lines, dtype=np.uint64)
        y = m * x + const
        y = np.tile(y, (end_of_frames_idx.shape[0], 1))
        y[1:, :] -= y[1, 0]
        y = np.concatenate((y, np.zeros_like(y)), axis=1)
        return y, m

    def __diff_vec_analysis_mscan(self, y: np.ndarray, lines: pd.Series,
                                  delta: np.uint64) -> pd.Series:
        diff_vec = np.abs(np.subtract(y, lines, dtype=np.int64))
        new_lines = np.concatenate((lines, np.zeros(len(lines) // 2, dtype=np.uint64)))
        missing_val = np.where(diff_vec > delta / 20)[0]
        while missing_val.shape[0] > 0:
            new_lines = np.concatenate((new_lines[:missing_val[0]],
                                        y[missing_val[0]],
                                        new_lines[missing_val[0]:]))
            missing_val = np.where(diff_vec > delta / 20)[0]
        return pd.Series(new_lines)

    def __finalize_lines_mscan(self, lines, delta):
        """ Sample the lines so that they're "silent" between frames """
        lines_between_frames = int(np.rint(self.frame_delay / delta))
        start_of_frame_idx = np.arange(start=0, stop=len(lines),
                                       step=self.num_of_lines + lines_between_frames,
                                       dtype=np.uint64)
        end_of_frame_idx = start_of_frame_idx + self.num_of_lines
        exact_lines = [lines[slice(start, end)] for start, end in zip(start_of_frame_idx, end_of_frame_idx)]
        exact_lines = np.array(list(chain.from_iterable(exact_lines)), dtype=np.uint64)
        self.num_of_frames = len(end_of_frame_idx)

        return pd.Series(exact_lines, dtype=np.uint64)

    def __diff_mat_analysis_si(self, y: np.ndarray, lines_mat: np.ndarray, last_idx: np.ndarray,
                               delta: int, rel_idx: np.ndarray) -> np.ndarray:
        """
        Check for missing\extra lines in the matrix of lines
        :param diff_mat: np.ndarray
        :param last_idx: last index of relevant lines in the frame
        :return:
        """
        diff_mat = np.abs(np.subtract(lines_mat, y, dtype=np.int64))
        missing_vals_rows, missing_vals_cols = np.where(diff_mat > delta / 20)
        frames_to_correct = np.where(last_idx != self.num_of_lines)[0]
        for frame_num in frames_to_correct:
            num_of_missing_lines = last_idx[frame_num] - self.num_of_lines
            if num_of_missing_lines < 0:
                for miss in missing_vals_cols[missing_vals_rows == frame_num]:
                    lines_mat[frame_num, :] = np.concatenate((lines_mat[frame_num, :miss],
                                                              np.atleast_1d(y[frame_num, miss]),
                                                              lines_mat[frame_num, miss:-1]))
            elif num_of_missing_lines > 0:
                cur_missing_cols = missing_vals_cols.copy()
                while cur_missing_cols.shape[0] > 0:
                    lines_mat[frame_num, :] = np.concatenate((lines_mat[frame_num, :missing_vals_cols[0]],
                                                              np.atleast_1d(y[frame_num, missing_vals_cols[0]]),
                                                              lines_mat[frame_num, missing_vals_cols[0]:-1]))
                    diff_line = np.abs(np.subtract(lines_mat[frame_num, :], y[frame_num, :], dtype=np.int64))
                    cur_missing_cols = np.where(diff_line > delta / 20)[0]

        return lines_mat

    def __finalize_lines_si(self, lines_mat: np.ndarray) -> pd.Series:
        """ Add back the start-of-frame times to the lines """
        frames_mat = np.array(self.dict_of_data['Frames']).reshape((len(self.dict_of_data['Frames']), 1))
        frames_mat = np.tile(frames_mat, (1, self.num_of_lines))
        lines_mat = lines_mat[:, :self.num_of_lines] + frames_mat
        return pd.Series(lines_mat.ravel())

    def __corrupt(self) -> Tuple[Dict, int]:
        """
        Data is corrupted, and no frame channel can help us.
        :return:
        """
        line_delta = self.__bins_bet_lines()
        self.dict_of_data['Lines'] = self.__extrapolate_line_data(line_point=self.dict_of_data['Lines'].at[0, 'abs_time'])
        return self.dict_of_data, line_delta

    def __rebuild(self) -> Tuple[Dict, int]:
        """
        Data is corrupted, but we can rebuild lines on top of the frame channel
        :return:
        """
        line_array = self.__create_line_array()
        self.dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        line_delta = self.last_event_time / (self.num_of_lines * int(self.num_of_frames))
        return self.dict_of_data, line_delta

    def __add_zeroth_line_event(self, lines: pd.Series) -> pd.Series:
        """
        If lines was used as the starting signal - add a 0 line time
        :param lines:
        :return: Corrected line signal
        """
        line_delta = lines.diff().median()
        zeroth_line_delta = np.abs(lines.iat[0] - line_delta) / line_delta
        if zeroth_line_delta < 0.05:  # first line came exactly line_delta time after the start of the experiment
            lines = pd.Series([0], dtype=np.uint64).append(lines)
            self.dict_of_data['Lines'] = pd.DataFrame([[0] * len(self.data_to_grab)],
                                                 columns=self.data_to_grab,
                                                 dtype=np.uint64)\
                .append(self.dict_of_data['Lines'], ignore_index=True)

        return lines

    def __si_valid(self) -> Tuple[Dict, int]:
        """
        Data is valid. Check whether we need a 0-time line event
        """
        line_delta = self.dict_of_data['Lines'].loc[:, 'abs_time'].diff().median()
        # zeroth_line_delta = np.abs(self.dict_of_data['Lines'].loc[0, 'abs_time'] - line_delta)/line_delta
        # if zeroth_line_delta < 0.05:
        #     self.dict_of_data['Lines'] = pd.DataFrame([[0] * len(self.data_to_grab)],
        #                                          columns=self.data_to_grab,
        #                                          dtype='uint64')\
        #         .append(self.dict_of_data['Lines'], ignore_index=True)
        return self.dict_of_data, np.uint64(line_delta)

    def __from_scratch(self) -> Tuple[Dict, int]:
        line_array = self.__create_line_array()
        self.dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        line_delta = self.last_event_time/(self.num_of_lines * int(self.num_of_frames))
        return self.dict_of_data, line_delta

    def __sweeps_rebuild(self) -> Tuple[Dict, int]:
        line_delta = 1 / self.line_freq if not self.bidir else 1 / (2 * self.line_freq)
        self.num_of_frames = np.ceil(self.last_event_time / (line_delta / self.binwidth * self.num_of_lines))
        line_array = self.__create_line_array(line_delta=line_delta/self.binwidth)
        self.dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        return self.dict_of_data, int(line_delta / self.binwidth)

    def __sweeps_from_scratch(self) -> Tuple[Dict, int]:
        sweep_vec = np.arange(self.max_sweep + 1, dtype=np.uint64)
        if len(sweep_vec) < 2:
            warnings.warn("All data was registered to a single sweep. Line data will be completely simulated.")
        else:
            self.dict_of_data['Lines'] = pd.DataFrame(
                sweep_vec * self.total_sweep_time,
                columns=['abs_time'], dtype=np.uint64)
        return self.dict_of_data, self.total_sweep_time

    def __extrapolate_line_data(self, line_point: int=0) -> pd.DataFrame:
        """
        From a single line signal extrapolate the presumed line data vector. The line frequency is doubled
        the original frequency. If needed, events will be discarded later.
        :param line_point: Start interpolation from this point.
        :return: pd.DataFrame of line data
        """
        # Verify input
        if line_point / self.line_delta > self.num_of_lines:  # The first recorded line came very late
            line_point = 0

        # Create the matrix containing the duplicate frame data
        delay_between_frames_in_bins = int(self.delay_between_frames / self.binwidth)
        time_of_frame = self.line_delta * self.num_of_lines \
                        + delay_between_frames_in_bins
        num_of_frames = max(int(np.floor(self.last_event_time / time_of_frame)), 1)
        time_of_frame_mat = np.tile(time_of_frame * np.arange(num_of_frames), (self.num_of_lines, 1))

        # Create the matrix containing the duplicate line data
        line_vec = np.arange(start=line_point, stop=self.line_delta * self.num_of_lines, step=self.line_delta,
                             dtype=np.uint64)
        line_vec = np.r_[np.flip(np.arange(start=line_point, stop=0, step=-self.line_delta,
                                           dtype=np.uint64)[1:], axis=0), line_vec]
        # Check if 0 should be included
        if line_vec[0] - self.line_delta == 0:
            line_vec = np.r_[0, line_vec]

        line_mat = np.tile(line_vec.reshape(len(line_vec), 1), (1, num_of_frames))

        # Add them up
        assert len(line_mat) > 0
        assert len(time_of_frame_mat) > 0
        final_mat = line_mat + time_of_frame_mat
        line_vec_final = np.ravel(final_mat, order='F')

        return pd.DataFrame(line_vec_final, columns=['abs_time'], dtype=np.uint64)

    def __create_line_array(self, line_delta: int=0) -> np.ndarray:
        """Create a pandas Series of start-of-line times"""

        total_lines = self.num_of_lines * int(self.num_of_frames)
        if line_delta == 0:
            line_array = np.arange(start=0, stop=self.last_event_time,
                                   step=self.last_event_time / total_lines,
                                   dtype=np.uint64)
        else:
            line_array = np.arange(start=0, stop=self.last_event_time,
                                   step=line_delta, dtype=np.uint64)
        return line_array

    def __validate_frame_input(self):

        if 'Frames' in self.dict_of_data.keys():
            self.dict_of_data['Frames'] = pd.DataFrame([[0] * len(self.data_to_grab)],
                                                       columns=self.data_to_grab,
                                                       dtype='uint64')\
                        .append(self.dict_of_data['Frames'], ignore_index=True)
        else:
            frame_array = self.__create_frame_array(lines=self.dict_of_data['Lines'].loc[:, 'abs_time'])
            self.dict_of_data['Frames'] = pd.DataFrame(frame_array, columns=['abs_time'], dtype='uint64')

        return self.dict_of_data


    def __create_frame_array(self, lines: pd.Series=None) -> np.ndarray:
        """Create a pandas Series of start-of-frame times"""

        num_of_recorded_lines = lines.shape[0]
        actual_num_of_frames = max(num_of_recorded_lines // self.num_of_lines, 1)

        if num_of_recorded_lines < self.num_of_lines:
            array_of_frames = np.linspace(start=0, stop=self.last_event_time, num=int(actual_num_of_frames),
                                          endpoint=False, dtype=np.uint64)
        else:
            unnecess_lines = num_of_recorded_lines % self.num_of_lines
            array_of_frames = lines.iloc[0 : int(num_of_recorded_lines-unnecess_lines) : self.num_of_lines]

        return np.array(array_of_frames)

    def __validate_created_data_channels(self) -> None:
        """
        Make sure that the dictionary that contains all data channels makes sense.
        """
        assert {'PMT1', 'Lines', 'Frames'} <= set(self.dict_of_data.keys())  # A is subset of B

        if self.dict_of_data['Frames'].shape[0] > self.dict_of_data['Lines'].shape[0]:  # more frames than lines
            raise UserWarning('More frames than lines, consider replacing the two.')

        try:
            if self.dict_of_data['TAG Lens'].shape[0] < self.dict_of_data['Lines'].shape[0]:
                raise UserWarning('More lines than TAG pulses, consider replacing the two.')
        except KeyError:
            pass

        try:
            if self.dict_of_data['Laser'].shape[0] < self.dict_of_data['Lines'].shape[0] or \
               self.dict_of_data['Laser'].shape[0] < self.dict_of_data['Frames'].shape[0]:
                raise UserWarning('Laser pulses channel contained less ticks than the Lines or Frames channel.')
        except KeyError:
            pass

        try:
            if self.dict_of_data['Laser'].shape[0] < self.dict_of_data['TAG Lens'].shape[0]:
                raise UserWarning('Laser pulses channel contained less ticks than the TAG lens channel.')
        except KeyError:
            pass

    def __validate_laser_input(self, pulses) -> pd.Series:
        """
        Create an orderly laser pulse train.
        :param pulses: Laser data
        :return:
        """
        import warnings

        diffs = pulses.loc[:, 'abs_time'].diff()
        rel_idx = (diffs <= 1.05*np.ceil((1 / (self.laser_freq * self.binwidth)))) & \
                  (diffs >= 0.95*np.floor((1 / (self.laser_freq * self.binwidth))))
        pulses_final = pulses[rel_idx]  # REMINDER: Laser offset wasn't added
        if len(pulses_final) < 0.9 * len(pulses):  # TODO: If there's holdafter time, there's a good chance that the pulses
                                                   # will not be periodic due to this extra time.
            warnings.warn("More than 10% of pulses were filtered due to bad timings. Make sure the laser input is fine.")

        pulses_final = pd.concat([pulses.loc[:0, :], pulses_final])  # Add back the first pulse
        pulses_final.reset_index(drop=True, inplace=True)

        return pulses_final

    def __add_phase_to_bidir_lines(self, lines: pd.Series):
        """
        "Fix" temporarily the lines for them to pass the corruption check
        :param lines: Lines after the phase change
        :param bidir_phase:
        :param binwidth:
        :return: Lines "pre-fix" of phase change - their original form
        """
        phase_in_seconds = self.bidir_phase * 1e-6
        if phase_in_seconds < 0:
            lines.iloc[1::2] -= np.uint64(np.abs(phase_in_seconds / self.binwidth))
        else:
            lines.iloc[1::2] += np.uint64(phase_in_seconds / self.binwidth)
        return lines
