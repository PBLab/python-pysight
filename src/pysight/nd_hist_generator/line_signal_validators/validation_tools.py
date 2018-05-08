"""
__author__ = Hagai Hargil
"""
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import warnings
import attr
from attr.validators import instance_of
from enum import Enum
from pysight.tkinter_gui_multiscaler import ImagingSoftware
from ..line_signal_validators.scanimage import ScanImageLineValidator
from ..line_signal_validators.mscan import MScanLineValidator


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
    Inputs:
        :param dict_of_data dict: Dictionary with keys corresponding to signal type,
        and values as DataFrames.
        :param data_to_grab list: Column names from the DF of the data to parse.
        :param num_of_lines int: Number of lines (x pixels) in the frame)
        :param num_of_frames int: Number of frames of the data. If "Frames" is not in the
        dict_of_data keys then this number is disregarded.
        :param binwidth float: Multiscaler binwidth in seconds.
        :param line_freq float: Line frequency of the fast scanner (x) in Hz.
        :param bidir bool: Whether the scan was bidirectional.
        :param delay_between_frames float: Time between subsequent frames. Depends on imaging
        software and hardware.
        :param use_sweeps bool: Use the sweeps of the multiscaler as an indicator of new lines.
        :bidir phase float: A number in microseconds typically reported by the scanning software.
        Helps with alignment of the generated image when using a bidirectional scan.
        :param imaging_software ImagingSoftware: Software used for imaging.
        :param change_thresh float: Percent-like [0, 1] threshold to discard irregular line signals.
        Above which PySight will terminate with an exception.
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
    change_thresh = attr.ib(default=0.3, validator=instance_of(float))
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
            LinesType.SCANIMAGE: self.__si,
            LinesType.MSCAN: self.__mscan,

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

    def __bins_bet_lines(self) -> int:
        """
        Calculate the line difference in bins, according to the frequency
        of the line signal.
        :param line_freq: Frequency of lines in Hz. Regardless of bidirectional scanning.
        :return: int - line delta
        """
        freq_in_bins = 1 / (self.line_freq * self.binwidth)
        return int(freq_in_bins / 2) if self.bidir else freq_in_bins

    def __calc_last_event_time(self) -> int:
        """
        Find the last event time for the experiment. Logic as follows:
        No lines \ frames data given: Last event time is the last photon time.
        Only lines data given: The last start-of-frame time is created, and the difference between subsequent frames
        in the data is added.
        Frames data exists: The last frame time plus the difference between subsequent frames is the last event time.
        :return int: Last event time
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

            # if num_of_lines_recorded > (self.num_of_lines * div):  # excessive number of lines
            #     last_line_of_last_frame = self.dict_of_data['Lines'].loc[:, 'abs_time']\
            #         .iloc[div * self.num_of_lines - 1]
            #     line_diff = int(self.dict_of_data['Lines'].iloc[:, 0].diff().mean())
            #     next_frame_event = self.dict_of_data['Lines'].iloc[div * self.num_of_lines, :]
            #     next_frame_event['abs_time'] += np.uint64(10 * line_diff)
            #     self.dict_of_data['Lines'] = self.dict_of_data['Lines'].loc[
            #                                  self.dict_of_data['Lines'].loc[:, 'abs_time'] <= last_line_of_last_frame,:]
            #     if self.image_soft == ImagingSoftware.SCANIMAGE.value:
            #         self.dict_of_data['Lines'] = self.dict_of_data['Lines'].append(next_frame_event)
            #     return int(last_line_of_last_frame + line_diff)

            if mod == 0:  # number of lines contained exactly in number of lines per frame
                return int(self.dict_of_data['Lines'].loc[:, 'abs_time'].iloc[-1] + self.dict_of_data['Lines']\
                    .loc[:, 'abs_time'].diff().mean())

            elif (num_of_lines_recorded < self.num_of_lines * (div+1)) and (div > 0):
                last_line_of_last_frame = self.dict_of_data['Lines'].loc[:, 'abs_time']\
                    .iloc[div * self.num_of_lines - 1]
                line_diff = int(self.dict_of_data['Lines'].iloc[:, 0].diff().mean())
                # missing_lines = self.num_of_lines - mod
                # next_frame_event = self.dict_of_data['Lines'].iloc[div * self.num_of_lines, :]
                # next_frame_event['abs_time'] += np.uint64(10 * line_diff)
                # if self.image_soft == ImagingSoftware.SCANIMAGE.value:
                #     self.dict_of_data['Lines'] = self.dict_of_data['Lines'].append(next_frame_event)
                #
                return int(last_line_of_last_frame + line_diff)

            elif div == 0:
                line_diff = int(self.dict_of_data['Lines'].iloc[:, 0].diff().mean())
                missing_lines = self.num_of_lines - num_of_lines_recorded + 1
                return self.dict_of_data['Lines'].iloc[-1, 0] + (line_diff * missing_lines)

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
            rel_idx = np.where(np.abs(lines.diff().pct_change(periods=1)) > self.change_thresh)[0]
            is_corrupt_sig: bool = (len(rel_idx) // 2) / lines.shape[0] > self.change_thresh
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

    def __si(self) -> Tuple[Dict, np.uint64]:
        return ScanImageLineValidator(sig_val=self).run()

    def __mscan(self) -> Tuple[Dict, np.uint64]:
        return MScanLineValidator(sig_val=self).run()

    def __corrupt(self) -> Tuple[Dict, int]:
        """
        Data is corrupted, and no frame channel can help us.
        :return:
        """
        self.line_delta = self.__bins_bet_lines()
        self.dict_of_data['Lines'] = self.__extrapolate_line_data(line_point=self.dict_of_data['Lines'].at[0, 'abs_time'])
        return self.dict_of_data, self.line_delta

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
        From a single line signal extrapolate the presumed line data vector. The line frequency is double
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
            if self.image_soft == 'MScan':
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

    def add_phase_to_bidir_lines(self, lines: pd.Series):
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
