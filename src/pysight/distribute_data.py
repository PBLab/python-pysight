"""
__author__ = Hagai Hargil
"""
import pandas as pd
import numpy as np
import attr
from typing import Dict, Tuple, List
from pysight.validation_tools import validate_line_input, validate_frame_input, \
    validate_laser_input, validate_created_data_channels, \
    calc_last_event_time
from attr.validators import instance_of


@attr.s(slots=True)
class DistributeData:
    """
    Separates the channel-specific data to their own channels
    # ABOUT TO BE REFACTORED
    """
    df = attr.ib(validator=instance_of(pd.DataFrame))
    dict_of_inputs = attr.ib(validator=instance_of(dict))
    x_pixels = attr.ib(default=512, validator=instance_of(int))
    y_pixels = attr.ib(default=512, validator=instance_of(int))
    num_of_frames = attr.ib(default=1, validator=instance_of(int))
    laser_freq = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    use_tag_bits = attr.ib(default=False, validator=instance_of(bool))
    use_sweeps = attr.ib(default=False, validator=instance_of(bool))
    time_after_sweep = attr.ib(default=int(96), validator=instance_of(int))
    acq_delay = attr.ib(default=int(0), validator=instance_of(int))
    line_freq = attr.ib(default=7900.0, validator=instance_of(float))
    bidir = attr.ib(default=False, validator=instance_of(bool))
    bidir_phase = attr.ib(default=-2.79, validator=instance_of(float))
    num_of_channels = attr.ib(default=3, validator=instance_of(int))
    data_range = attr.ib(default=1, validator=instance_of(int))
    line_delta = attr.ib(init=False)
    dict_of_data = attr.ib(init=False)
    data_to_grab = attr.ib(init=False)

    def run(self) -> None:
        """
        Main pipeline
        :return:
        """
        self.dict_of_data = self.determine_data_channels()

    @property
    def total_sweep_time(self):
        return self.acq_delay + self.data_range + self.time_after_sweep

    def __allocate_data_by_channel(self):
        """
        Go over the channels and find the events from that specific channel, assigning
        them to a dictionary with a suitable name.
        :return: Dict containing the data
        """
        dict_of_data = {}
        self.data_to_grab = ['abs_time', 'sweep']
        if self.use_tag_bits:
            self.data_to_grab.extend(['tag', 'edge'])
        for key in self.dict_of_inputs:
            relevant_values = self.df.loc[self.df['channel'] == self.dict_of_inputs[key], self.data_to_grab]
            # NUMBA SORT NOT WORKING:
            # sorted_vals = numba_sorted(relevant_values.values)
            # dict_of_data[key] = pd.DataFrame(sorted_vals, columns=['abs_time'])
            if key in ['PMT1', 'PMT2']:
                dict_of_data[key] = relevant_values.reset_index(drop=True)
                dict_of_data[key]['Channel'] = 1 if 'PMT1' == key else 2  # Channel is the spectral channel
            else:
                dict_of_data[key] = relevant_values.sort_values(by=['abs_time']).reset_index(drop=True)

        return dict_of_data

    def determine_data_channels(self) -> Dict:
        """ Create a dictionary that contains the data in its ordered form."""

        if self.df.empty:
            raise ValueError('Received dataframe was empty.')

        dict_of_data = self.__allocate_data_by_channel()

        if 'Frames' in dict_of_data:
            self.num_of_frames = dict_of_data['Frames'].shape[0] + 1  # account for first frame

        if self.bidir:
            dict_of_data['Lines'] = self.__add_phase_to_bidir_lines(dict_of_data['Lines'])

        # Validations
        last_event_time = calc_last_event_time(dict_of_data=dict_of_data, lines_per_frame=self.y_pixels)
        dict_of_data, self.line_delta = validate_line_input(dict_of_data=dict_of_data, num_of_lines=self.y_pixels,
                                                            num_of_frames=self.num_of_frames, line_freq=self.line_freq,
                                                            last_event_time=last_event_time, bidir=self.bidir,
                                                            cols_in_data=self.data_to_grab, use_sweeps=self.use_sweeps,
                                                            max_sweep=self.df['sweep'].max(),
                                                            bidir_phase=self.bidir_phase,
                                                            total_sweep_time=self.total_sweep_time)
        dict_of_data = validate_frame_input(dict_of_data=dict_of_data,
                                            num_of_lines=self.y_pixels, binwidth=self.binwidth,
                                            last_event_time=last_event_time,
                                            cols_in_data=self.data_to_grab)
        try:
            dict_of_data['Laser'] = validate_laser_input(dict_of_data['Laser'], laser_freq=self.laser_freq,
                                                         binwidth=self.binwidth)
        except KeyError:
            pass

        validate_created_data_channels(dict_of_data)
        return dict_of_data

    def __add_phase_to_bidir_lines(self, lines: pd.DataFrame) -> pd.DataFrame:
        """
        Add the phase to all returning-phase lines
        :param lines: pd.DataFrame of the line signals
        :return: pd.DataFrame
        """
        phase_in_seconds = self.bidir_phase * 1e-6
        if phase_in_seconds < 0:
            lines.abs_time.iloc[1::2] -= np.uint64(np.abs(phase_in_seconds / self.binwidth))
        else:
            lines.abs_time.iloc[1::2] += np.uint64(phase_in_seconds / self.binwidth)
        return lines
