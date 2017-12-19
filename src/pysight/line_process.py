"""
__author__ = Hagai Hargil
"""
import numpy as np
import pandas as pd
import attr
from attr.validators import instance_of
from pysight.validation_tools import LineDataType
import warnings


@attr.s(slots=True)
class LineProcess:
    """
    A switch class to process the line signal
    """
    line_freq = attr.ib(default=7930., validator=instance_of((int, float)))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    bidir = attr.ib(default=False, validator=instance_of(bool))
    last_event_time = attr.ib(default=1, validator=instance_of(int))
    dict_of_data = attr.ib(default={}, validator=instance_of(dict))
    num_of_lines = attr.ib(default=1, validator=instance_of(int))
    delay_between_frames = attr.ib(default=0.0011355, validator=instance_of(float))
    num_of_frames = attr.ib(default=1, validator=instance_of(int))
    cols_in_data = attr.ib(default=None, validator=instance_of(list))
    max_sweep = attr.ib(default=np.uint64(0), validator=instance_of(np.uint64))
    total_sweep_time = attr.ib(default=100, validator=instance_of(int))
    bidir_phase = attr.ib(default=-2.6, validator=instance_of(float))

    @property
    def __choice_table(self):
        return {
            LineDataType.CORRUPT: self.__corrupt,
            LineDataType.REBUILD: self.__rebuild,
            LineDataType.VALID: self.__valid,
            LineDataType.FROM_SCRATCH: self.__from_scratch,
            LineDataType.SWEEPS_REBUILD: self.__sweeps_rebuild,
            LineDataType.SWEEPS_FROM_SCRATCH: self.__sweeps_from_scratch,
            LineDataType.ADD: self.__add
        }

    def __corrupt(self):
        # Data is corrupted, and no frame channel can help us.
        line_delta = self.__bins_bet_lines()
        self.dict_of_data['Lines'] = self.__extrapolate_line_data(
            line_delta=line_delta,
            line_point=self.dict_of_data['Lines'].at[0, 'abs_time'])
        return line_delta

    def __rebuild(self):
        # Data is corrupt, but we can rebuild lines on top of the frame channel
        line_array = self.__create_line_array()
        self.dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        line_delta = self.last_event_time / (self.num_of_lines * int(self.num_of_frames))
        return line_delta

    def __valid(self):
        # Data is valid. Check whether we need a 0-time line event
        line_delta = self.dict_of_data['Lines'].loc[:, 'abs_time'].diff().mean()
        zeroth_line_delta = np.abs(self.dict_of_data['Lines'].loc[0, 'abs_time'] - line_delta)/line_delta
        if zeroth_line_delta < 0.05:
            self.dict_of_data['Lines'] = pd.DataFrame([[0] * len(self.cols_in_data)],
                                                      columns=self.cols_in_data,
                                                      dtype='uint64')\
                .append(self.dict_of_data['Lines'], ignore_index=True)
        return np.uint64(line_delta)

    def __from_scratch(self):
        line_array = self.__create_line_array()
        self.dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        line_delta = self.last_event_time/(self.num_of_lines * int(self.num_of_frames))
        return line_delta

    def __sweeps_rebuild(self):
        line_delta = 1 / self.line_freq if not self.bidir else 1 / (2 * self.line_freq)
        self.num_of_frames = np.ceil(self.last_event_time / (line_delta / self.binwidth * self.num_of_lines))
        line_array = self.__create_line_array(line_delta=line_delta/self.binwidth)
        self.dict_of_data['Lines'] = pd.DataFrame(line_array, columns=['abs_time'], dtype='uint64')
        return int(line_delta / self.binwidth)

    def __sweeps_from_scratch(self):
        sweep_vec = np.arange(self.max_sweep + 1, dtype=np.uint64)
        if len(sweep_vec) < 2:
            warnings.warn("All data was registered to a single sweep. Line data will be completely simulated.")
        else:
            self.dict_of_data['Lines'] = pd.DataFrame(
                sweep_vec * self.total_sweep_time,
                columns=['abs_time'], dtype=np.uint64)
        return self.total_sweep_time

    def __add(self):
        change_thresh = 0.5
        first_idx = self.dict_of_data['Lines'].abs_time[
            self.dict_of_data['Lines'].abs_time.diff().pct_change(periods=1) > change_thresh]\
            .diff().index[0]  # Number of lines in the first frame

        num_of_columns = len(self.dict_of_data['Lines'].columns)
        if first_idx < self.num_of_lines:  # add fake lines
            if self.dict_of_data['Lines'].abs_time.iloc[0] < 100:  # rough assessment so that we won't have non-unique lines
                start_time = self.dict_of_data['Lines'].abs_time.iloc[0] + 1
                end_time = self.dict_of_data['Lines'].abs_time.iloc[1]
            else:
                start_time = 0
                end_time = self.dict_of_data['Lines'].abs_time.iloc[0]
            new_df = pd.DataFrame([[0] * num_of_columns] * (self.num_of_lines-first_idx),
                                  columns=self.dict_of_data['Lines'].columns,
                                  dtype=np.uint64)
            new_df.abs_time += np.linspace(start=start_time, stop=end_time, num=self.num_of_lines-first_idx,
                                           dtype=np.uint64, endpoint=False)
            line_delta = self.dict_of_data['Lines'].loc[:first_idx-1, 'abs_time'].diff().mean()
            self.dict_of_data['Lines'] = new_df.append(self.dict_of_data['Lines']).reset_index(drop=True)

        elif first_idx > self.num_of_lines:
            self.dict_of_data['Lines'] = self.dict_of_data['Lines']\
                [first_idx - self.num_of_lines:, :].reset_index(drop=True)
            line_delta = self.dict_of_data['Lines'].loc[first_idx+1:, 'abs_time'].diff().mean()

        elif first_idx == self.num_of_lines:
            line_delta = self.dict_of_data['Lines'].loc[:self.num_of_lines-1, 'abs_time'].diff().mean()

        return line_delta

    def process(self, case):
        """
        Choose the right case for the job
        :param case:
        :return:
        """
        assert isinstance(case, LineDataType)

        line_delta = self.__choice_table[case]()
        return self.dict_of_data, line_delta

    def __bins_bet_lines(self):
        """
        Calculate the line difference in bins, according to the frequency
        of the line signal.
        :return: int - line delta
        """
        freq_in_bins = 1 / (self.line_freq * self.binwidth)
        return int(freq_in_bins / 2) if self.bidir else freq_in_bins

    def __extrapolate_line_data(self, line_delta, line_point):
        """
        From a single line signal extrapolate the presumed line data vector. The line frequency is doubled
        the original frequency. If needed, events will be discarded later.
        :param line_delta: Bins between subsequent lines.
        :param line_point: Start interpolation from this point.
        :return: pd.DataFrame of line data
        """
        # Verify input
        if line_point / line_delta > self.num_of_lines:  # The first recorded line came very late
            line_point = 0

        # Create the matrix containing the duplicate frame data
        delay_between_frames_in_bins = int(self.delay_between_frames / self.binwidth)
        time_of_frame = line_delta * self.num_of_lines + delay_between_frames_in_bins
        num_of_frames = max(int(np.floor(self.last_event_time / time_of_frame)), 1)
        time_of_frame_mat = np.tile(time_of_frame * np.arange(num_of_frames), (self.num_of_lines, 1))

        # Create the matrix containing the duplicate line data
        line_vec = np.arange(start=line_point, stop=line_delta * self.num_of_lines, step=line_delta,
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

    def __create_line_array(self, line_delta: float=0.) -> np.ndarray:
        """Create a pandas Series of start-of-line times"""

        if (self.last_event_time is None) or (self.num_of_lines is None) or (self.num_of_frames is None):
            raise ValueError('Wrong input detected.')

        if (self.num_of_lines <= 0) or (self.num_of_frames <= 0):
            raise ValueError('Number of lines and frames has to be positive.')

        if self.last_event_time <= 0:
            raise ValueError('Last event time is zero or negative.')

        total_lines = self.num_of_lines * int(self.num_of_frames)
        if 0.0 == line_delta:
            line_array = np.arange(start=0, stop=self.last_event_time,
                                   step=self.last_event_time / total_lines, dtype=np.uint64)
        else:
            line_array = np.arange(start=0, stop=self.last_event_time, step=line_delta, dtype=np.uint64)
        return line_array
