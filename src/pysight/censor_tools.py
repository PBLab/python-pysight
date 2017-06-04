"""
__author__ = Hagai Hargil
"""
import attr
import numpy as np
import pandas as pd
from attr.validators import instance_of
from pysight.movie_tools import Volume
from collections import deque, namedtuple


@attr.s(slots=True)
class CensorCorrection(object):
    df = attr.ib(validator=instance_of(pd.DataFrame))
    movie = attr.ib()
    reprate = attr.ib(validator=instance_of(float))
    binwidth = attr.ib(validator=instance_of(float))
    offset = attr.ib(validator=instance_of(int))
    all_laser_pulses = attr.ib()

    def create_laser_pulses_deque(self) -> np.ndarray:
        """
        If data has laser pulses - return them. Else - simulate them with an offset
        """
        laser_pulses_deque = deque()
        pulse_grid = namedtuple('PulseGrid', ('x_pulses', 'y_pulses'))
        start_time = 0
        step = int(np.ceil(1 / (self.reprate * self.binwidth)))
        volumes_in_movie = self.movie.gen_of_volumes()

        if self.all_laser_pulses == 0:  # no 'Laser' data was recorded
            for vol in volumes_in_movie:
                x_pulses = np.arange(start=start_time+self.offset, stop=vol.end_time,
                                     step=step, dtype=np.uint64)
                y_pulses = np.arange(start=start_time+self.offset, stop=vol.metadata['Y'].end+step,
                                     step=step)
                grid = pulse_grid(x_pulses, y_pulses)
                laser_pulses_deque.append(grid)
        else:
            for vol in volumes_in_movie:
                x_pulses = self.all_laser_pulses[(self.all_laser_pulses >= vol.abs_start_time-step) &
                                                   (self.all_laser_pulses <= vol.end_time+step)] + self.offset
                y_pulses = 1
                grid = pulse_grid(x_pulses, y_pulses)
                laser_pulses_deque.append(grid)
        return laser_pulses_deque

    def get_bincount_deque(self):
        bincount_deque = deque()
        laser_pulses_deque = self.create_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=laser_pulses_deque[idx].x_pulses)
            bincount_deque.append(censored.gen_bincount())
        return bincount_deque

    def find_temporal_structure_deque(self):
        temp_struct_deque = deque()
        laser_pulses_deque = self.create_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=laser_pulses_deque[idx].x_pulses,
                                      binwidth=self.binwidth, reprate=self.reprate)
            temp_struct_deque.append(censored.find_temp_structure())
        return temp_struct_deque

    def gen_array_of_hists_deque(self) -> np.ndarray:
        """
        Go through each volume in the deque and find the laser pulses for each pixel, creating a summed histogram per pixel.
        :return:
        """
        temp_struct_deque = deque()
        laser_pulses_deque = self.create_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=laser_pulses_deque[idx].x_pulses,
                                      binwidth=self.binwidth, reprate=self.reprate)
            temp_struct_deque.append(censored.gen_array_of_hists())

@attr.s(slots=True)
class CensoredVolume(object):
    df = attr.ib(validator=instance_of(pd.DataFrame))
    vol = attr.ib(validator=instance_of(Volume))
    laser_pulses = attr.ib(validator=instance_of(np.ndarray))
    offset = attr.ib(validator=instance_of(int))
    binwidth = attr.ib(default=800e-12)
    reprate = attr.ib(default=80e6)

    def gen_bincount(self) -> np.ndarray:
        """
        Bin the photons into their relative laser pulses, and count how many photons arrived due to each pulse.
        """
        hist, _ = np.histogram(self.df['abs_time'].values, bins=self.laser_pulses)
        return np.bincount(hist)

    def find_temp_structure(self) -> np.ndarray:
        """
        Generate a summed histogram of the temporal structure of detected photons.
        """
        bins = np.arange(start=0, stop=np.ceil(1/(self.reprate * self.binwidth)) + 1, step=1)
        subtracted_times, _, _ = self.sort_photons_in_pulses()
        hist, _ = np.histogram(subtracted_times, bins=bins)
        return hist

    def sort_photons_in_pulses(self):
        """
        Helper function to generate a searchsorted output of photons in laser pulses.
        """
        pulses = self.laser_pulses
        sorted_indices: np.ndarray = np.searchsorted(pulses, self.df['abs_time'].values) - 1
        array_of_laser_starts = pulses[sorted_indices]
        subtracted_times = self.df['abs_time'].values - array_of_laser_starts
        return subtracted_times, array_of_laser_starts, sorted_indices

    def gen_array_of_hists(self) -> np.ndarray:
        """
        For a specific frame, sort photons and laser pulses inside the pixels to gain
        statistics on the distribution of the photons inside the pixels.
        :return: np.ndarray of the same size as the original image. Each pixels contains
        a histogram inside it.
        """
        # hist, edges = self.vol.create_hist()
        # lines_in_vol = self.df.index.get_level_values('Lines').categories.values
        # sorted_pulses = np.searchsorted(lines_in_vol, self.laser_pulses) - 1
        # # Clean unneeded pulses (possibly first and last)
        # relevant_indices_for_pulses, = np.where(sorted_pulses > 0)
        # sorted_pulses = sorted_pulses[relevant_indices_for_pulses]
        # corrected_pulse_times = self.laser_pulses[relevant_indices_for_pulses]
        # relative_pulse_times = corrected_pulse_times - lines_in_vol[sorted_pulses]
        #
        # digitized_laser_x   = np.digitize(self.laser_pulses, bins=edges[0]) - 1
        # digitized_laser_y   = np.digitize(relative_pulse_times, bins=edges[1]) - 1
        # digitized_photons_x = np.digitize(self.df['abs_time'].values, bins=edges[0]) - 1
        # digitized_photons_y = np.digitize(self.df['time_rel_line'].values, bins=edges[1]) - 1
        #
        # image_bincount = np.zeros_like(hist, dtype=object)
        # for row in range(len(edges[0]) - 1):
        #     print('Row number {}'.format(row))
        #     pulses_row = self.laser_pulses[np.where(digitized_laser_x == row)[0]]
        #     photons_row = self.df['abs_time'].values[np.where(digitized_photons_x == row)[0]]
        #     # row_hist, _ = np.histogram(photons_row, bins=pulses_row)
        #     for col in range(len(edges[1]) - 1):
        #         print('Col number {}'.format(col))
        #         all_pulses_of_col = np.where(digitized_laser_y == col)[0]
        #         pulse_cross_section = all_pulses_of_col[(all_pulses_of_col >= pulse_idx_row.min()) &
        #                                                 (all_pulses_of_col <= pulse_idx_row.max())]
        #         if len(photons_idx_row) > 0:  # Photons arrived during this line
        #             all_photons_of_col = np.where(digitized_photons_y == col)[0]
        #             photon_cross_section = all_photons_of_col[(all_photons_of_col >= photons_idx_row.min()) &
        #                                                       (all_photons_of_col <= photons_idx_row.max())]
        #             col_hist, _ = np.histogram(self.df['abs_time'].values[photon_cross_section],
        #                                        bins=self.laser_pulses[pulse_cross_section])
        #         else:
        #             col_hist, _ = np.histogram(np.array([]), bins=self.laser_pulses[pulse_cross_section])
        #
        #         image_bincount[row, col] = np.bincount(col_hist)
        #
        #
        # return image_bincount
        pass
