"""
__author__ = Hagai Hargil
"""
import attr
from typing import List
import numpy as np
from numba import jit, uint8, int64, uint16, uint64
import pandas as pd
from attr.validators import instance_of


@attr.s(slots=True)
class CensorCorrection(object):
    df = attr.ib(validator=instance_of(pd.DataFrame))
    deque_of_vols = attr.ib()
    reprate = attr.ib(validator=instance_of(float))
    binwidth = attr.ib(validator=instance_of(float))
    offset = attr.ib(validator=instance_of(int))

    @property
    def start_time(self) -> int: return self.df['abs_time'].min()

    @property
    def end_time(self) -> int: return self.df['abs_time'].max() \
        + np.ceil(self.reprate * self.binwidth)

    @property
    def laser_pulses(self) -> np.ndarray:
        return np.arange(start=self.start_time + self.offset, stop=self.end_time,
                         step=np.ceil(1/(self.reprate * self.binwidth)), dtype=np.uint64)

    def gen_bincount(self) -> np.ndarray:
        """
        Bin the photons into their relative laser pulses, and count how many photons arrived due to each pulse.
        """
        hist, _ = np.histogram(self.df['abs_time'].values, bins=self.laser_pulses)
        return np.bincount(hist)

    def sort_photons_in_pulses(self):
        """
        Helper function to generate a searchsorted output of photons in laser pulses.
        """
        pulses = self.laser_pulses
        sorted_indices: np.ndarray = np.searchsorted(pulses, self.df['abs_time'].values) - 1
        array_of_laser_starts = pulses[sorted_indices]
        subtracted_times = self.df['abs_time'].values - array_of_laser_starts
        return subtracted_times, array_of_laser_starts, sorted_indices

    def gen_temp_structure(self) -> np.ndarray:
        """
        Generate a summed histogram of the temporal structure of detected photons.
        """
        bins = np.arange(start=0, stop=np.ceil(1/(self.reprate * self.binwidth)) + 1, step=1)
        subtracted_times, _, _ = self.sort_photons_in_pulses()
        hist, _ = np.histogram(subtracted_times, bins=bins)
        return hist

    def gen_array_of_hists(self) -> np.ndarray:
        """
        Go through each volume in the deque and find the laser pulses for each pixels, creating a summed histogram per pixel.
        :return:
        """
        subtracted_times, array_of_laser_starts, sorted_indices = self.sort_photons_in_pulses()
        pulses = self.laser_pulses
        for vol in self.deque_of_vols:
            cur_pulses = list(range(2))
            cur_pulses[0] = pulses[pulses >= vol.edges[0].min()]
            cur_pulses[0] = cur_pulses[cur_pulses < vol.edges[0].max()]

            cur_pulses[1] = pulses[pulses >= vol.edges[1].min()]
            cur_pulses[1] = cur_pulses[cur_pulses < vol.edges[1].max()]

            relevant_photons = self.df['abs_time'].values[sorted_indices == sorted_pulses_in_frame]

