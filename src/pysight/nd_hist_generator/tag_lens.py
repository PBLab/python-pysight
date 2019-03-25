import logging
import numpy as np
import pandas as pd
import warnings
from numba import jit, uint64
from typing import Tuple, List
import attr
from attr.validators import instance_of
from itertools import chain


@attr.s
class TagPipeline:
    """
    Pipeline to interpolate TAG lens pulses

    :param pd.DataFrame photons: DataFrame of photons in experiment
    :param pd.Series tag_pulses: Series of TAG events
    :param float freq: Expected frequency of TAG sync pulses in Hz
    :param float binwidth: Multiscaler binwidth in seconds (100 ps == 100e-12)
    :param int num_of_pulses: Number of TAG pulses from the driver per period. Currently NotImplemented
    :param bool to_phase: Whether to compensate for the sinusoidal pattern of the TAG. Leave True
    :param int offset: Offset in degrees given from the TAG driver for each pulse
    """

    photons = attr.ib(validator=instance_of(pd.DataFrame))
    tag_pulses = attr.ib(validator=instance_of(pd.Series))
    freq = attr.ib(default=189e3, validator=instance_of(float))  # Hz
    binwidth = attr.ib(
        default=800e-12, validator=instance_of(float)
    )  # Multiscaler binwidth
    num_of_pulses = attr.ib(
        default=1, validator=instance_of(int)
    )  # Number of pulses per TAG period
    to_phase = attr.ib(
        default=True, validator=instance_of(bool)
    )  # compensate the sinusoidial pattern of TAG
    offset = attr.ib(
        default=0, validator=instance_of(int)
    )  # offset the TAG phase [degrees]
    finished_pipe = attr.ib(init=False)

    @property
    def first_photon(self):
        return self.photons.abs_time.min().astype(np.uint64)

    @property
    def last_photon(self):
        return self.photons.abs_time.max().astype(np.uint64)

    def run(self):
        """ Main pipeline """
        clean_tag = self.__preserve_relevant_tag_pulses().reset_index(drop=True)
        verifier = TagPeriodVerifier(
            tag=clean_tag,
            freq=self.freq / self.num_of_pulses,
            binwidth=self.binwidth,
            first_photon=np.int64(self.first_photon),
            last_photon=self.last_photon,
        )
        verifier.verify()
        if verifier.success:
            phaser = TagPhaseAllocator(
                photons=self.photons,
                tag=verifier.tag,
                pulses_per_period=self.num_of_pulses,
                to_phase=self.to_phase,
                offset=self.offset,
            )
            phaser.allocate_phase()
            self.photons = phaser.photons
            self.finished_pipe = True
        else:
            self.finished_pipe = False
            try:  # Add the 'TAG' column to check data after pipeline finishes
                self.photons["TAG"] = np.pad(
                    clean_tag,
                    (self.photons.shape[0] - len(self.tag_pulses), 0),
                    "constant",
                )
            except ValueError:  # more TAG pulses than events
                self.photons["TAG"] = self.tag_pulses[: self.photons.shape[0]]

    def __preserve_relevant_tag_pulses(self):
        """ Keep only TAG pulses that are in the timeframe of the experiment """
        relevant_tag_pulses = (self.tag_pulses >= self.first_photon) & (
            self.tag_pulses <= self.last_photon
        )
        return self.tag_pulses.loc[relevant_tag_pulses]


@attr.s
class TagPeriodVerifier:
    """ Verify input to the TAG pipeline, and add missing pulses accordingly """

    tag = attr.ib(validator=instance_of(pd.Series))
    last_photon = attr.ib(validator=instance_of(np.uint64))
    freq = attr.ib(default=189e3, validator=instance_of(float))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    jitter = attr.ib(
        default=0.05, validator=instance_of(float)
    )  # Allowed jitter of signal, between 0 - 1
    first_photon = attr.ib(default=np.int64(0), validator=instance_of(np.int64))
    allowed_corruption = attr.ib(default=0.3, validator=instance_of(float))
    success = attr.ib(init=False)

    @property
    def period(self):
        return int(np.ceil(1 / (self.freq * self.binwidth)))

    @property
    def allowed_noise(self):
        return int(np.ceil(self.jitter * self.period))

    def verify(self):
        """ Main script to verify and correct the recorded TAG lens pulses """

        # Find the borders of the disordered periods
        start_idx, end_idx = self.__obtain_start_end_idx()
        # Add \ remove TAG pulses in each period
        if isinstance(start_idx, np.ndarray) and isinstance(end_idx, np.ndarray):
            self.__fix_tag_pulses(start_idx, end_idx)
            self.__add_last_event_manually()
            self.success = True
        else:
            self.success = False

    def __obtain_start_end_idx(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Create two vectors corresponding to the starts and ends of the 'wrong' periods of the TAG lens """
        diffs = self.tag.diff()
        diffs[0] = self.period
        logging.debug(
            "The mean frequency of TAG events is {:0.2f} Hz.".format(
                1 / (np.mean(diffs) * self.binwidth)
            )
        )
        delta = np.abs(diffs - self.period)
        diffs[delta < self.allowed_noise] = 0  # regular events
        diffs[diffs != 0] = 1
        if np.sum(diffs) / len(diffs) > self.allowed_corruption:
            logging.warning(
                f"Over {self.allowed_corruption * 100}% of TAG pulses were out-of-phase."
                " Stopping TAG interpolation."
            )
            return (-1, -1)

        diff_of_diffs = diffs.diff()  # a vec containing 1 at the start
        #                               of a "bad" period, and -1 at its end
        diff_of_diffs[0] = 0
        starts_and_stops = diff_of_diffs[diff_of_diffs != 0]
        start_idx = starts_and_stops[starts_and_stops == 1].index - 1
        end_idx = starts_and_stops[starts_and_stops == -1].index - 1
        return start_idx.values, end_idx.values

    def __fix_tag_pulses(self, starts: np.ndarray, ends: np.ndarray):
        """ Iterate over the disordered periods and add or remove pulses """

        if len(starts) == 0:  # Nothing fix
            return
        period = self.period
        items_to_discard = []
        start_iter_at = 0
        # If start contains a 0 - manually add TAG pulses
        if starts[0] == 0:
            start_iter_at = 1
            new_ser = pd.Series(
                np.arange(
                    start=self.tag[ends[0]] - period,
                    stop=self.first_photon - 1,
                    step=-period,
                    dtype=np.uint64,
                ),
                dtype=np.uint64,
            )
            self.tag = self.tag.append(new_ser, ignore_index=True).astype(np.uint64)
            items_to_discard.append(np.arange(starts[0], ends[0]))
        jitter = self.jitter
        new_data, returned_items_to_discard = numba_iterate_over_disordered(
            tag=self.tag.values,
            starts=starts[start_iter_at:],
            ends=ends[start_iter_at:],
            period=period,
            jitter=jitter,
        )
        flattened_items_to_discard = list(
            chain.from_iterable(items_to_discard + returned_items_to_discard)
        )
        self.tag.drop(flattened_items_to_discard, inplace=True)
        flattened_new_data = list(chain.from_iterable(new_data))
        self.tag = (
            self.tag.append(
                pd.Series(flattened_new_data, dtype=np.uint64), ignore_index=True
            )
            .sort_values()
            .reset_index(drop=True)
        )
        assert self.tag.dtype == np.uint64

    def __add_last_event_manually(self):
        """ Insert a 'fake' TAG event to encapsulate the last remaining photons """
        last_tag_val = self.tag.values[-1] + self.period
        self.tag = self.tag.append(
            pd.Series(last_tag_val, dtype=np.uint64), ignore_index=True
        )
        assert self.tag.dtype == np.uint64


@attr.s
class TagPhaseAllocator:
    """ Assign a phase to each photon """

    photons = attr.ib(validator=instance_of(pd.DataFrame))
    tag = attr.ib(validator=instance_of(pd.Series))
    pulses_per_period = attr.ib(default=1, validator=instance_of(int))
    to_phase = attr.ib(default=True, validator=instance_of(bool))
    offset = attr.ib(default=0, validator=instance_of(int))
    TAG_DRIVER_OFFSET = attr.ib(default=90, validator=instance_of(int))

    @property
    def offset_rad(self):
        """ Convert degrees to radians """
        return (self.offset + self.TAG_DRIVER_OFFSET) * np.pi / 180

    def allocate_phase(self):
        """ Using Numba functions allocate the proper phase to the photons """
        bin_idx, relevant_bins = numba_digitize(
            self.photons.abs_time.values, self.tag.values
        )
        relevant_bins[bin_idx >= len(self.tag)] = False
        photons = self.photons.abs_time.values.astype(float)
        photons[np.logical_not(relevant_bins)] = np.nan
        relevant_photons = np.compress(relevant_bins, photons)
        phase_vec = numba_find_phase(
            photons=relevant_photons,
            bins=np.compress(relevant_bins, bin_idx),
            raw_tag=self.tag.values,
            to_phase=self.to_phase,
            offset=self.offset_rad,
        )
        if not self.to_phase:
            phase_vec = phase_vec.astype(np.uint16)
        first_relevant_photon_idx = photons.shape[0] - relevant_photons.shape[0]
        photons[first_relevant_photon_idx:] = phase_vec
        self.photons["Phase"] = photons
        self.photons.dropna(how="any", inplace=True)

        assert self.photons["Phase"].any() >= -1
        assert self.photons["Phase"].any() <= 1


@jit(nopython=True, cache=True)
def numba_digitize(values: np.array, bins: np.array) -> np.array:
    """ Numba'd version of np.digitize. """
    bins = np.digitize(values, bins)
    relevant_bins = bins > 0
    return bins, relevant_bins


@jit(nopython=True, cache=True)
def numba_find_phase(
    photons: np.array, bins: np.array, raw_tag: np.array, to_phase: bool, offset: float
) -> np.array:
    """
    Find the phase [0, 2pi) of the photon for each event in `photons`.

    :return np.ndarray: Array with the size of photons containing the phases.
    """

    phase_vec = np.zeros_like(photons, dtype=np.float32)
    tag_diff = np.diff(raw_tag)

    for idx, cur_bin in enumerate(bins):  # values of indices that changed
        phase_vec[idx] = (photons[idx] - raw_tag[cur_bin - 1]) / tag_diff[cur_bin - 1]

    if to_phase:
        phase_vec_float = np.sin(phase_vec * 2 * np.pi + offset)
        return phase_vec_float.astype(np.float32)

    return phase_vec.astype(np.float32)


@jit(cache=True)
def numba_iterate_over_disordered(
    tag: np.ndarray, starts: np.ndarray, ends: np.ndarray, period: int, jitter: float
) -> Tuple[List, List]:
    """
    Numba'd version of the main TAG iteration.
    Currently not working in nopython due to some bugs with arange
    """
    new_data = []
    items_to_discard = []
    jitter_int = period * jitter
    row_idx = 1
    for start_idx, end_idx in zip(starts, ends):
        start_val = tag[start_idx]
        end_val = tag[end_idx]
        if np.abs(end_val - start_val) - period > jitter_int:
            l = np.arange(end_val - period, start_val, -period, np.uint64)
            new_data.append(l)
        items_to_discard.append(np.arange(start_idx + 1, end_idx))
        row_idx += 1
    return (new_data, items_to_discard)
