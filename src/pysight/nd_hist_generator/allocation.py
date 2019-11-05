import logging
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import attr
from attr.validators import instance_of
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pysight.nd_hist_generator.tag_lens import TagPipeline


@attr.s(slots=True)
class Allocate(object):
    """
    Allocate photons to their coordinates (frames, lines, etc.).

    :param pd.DataFrame df_photons: DataFrame with photons
    :param dict dict_of_data: Organized data parsed from file
    :param float laser_freq: Frequency of laser pulses in Hz
    :param float binwidth: Binwidth of multiscaler in seconds (100 ps == 100e-12)
    :param bool bidir: Whether the scan was performed in a bi-directional manner
    :param float tag_freq: TAG lens frequency in Hz
    :param int tag_pulses: Number of TAG driver pulses per period (currently ``NotImplemented``)
    :param float phase: Phase of the resonant scanner when used in a bi-directional mode
    :param bool keep_unidir: Whether to keep photons from the returning phase of the resonant scanner
    :param bool flim: Whether to perform FLIM analysis (currently ``NotImplemented``)
    :param dict exp_params: Parameters from the FLIM fit (currently ``NotImplemented``)
    :param bool censor: Whether to perform censor correction (currently ``NotImplemented``)
    :param bool tag_interp_ok: Whether the TAG lens interpolation process completed successfully
    :param bool tag_to_phase: Whether to interpolate the TAG lens sinusoidal pattern
    :param int tag_offset: Offset in degrees of the TAG pulse
    """

    # TODO: Variable documentation
    df_photons = attr.ib(validator=instance_of(pd.DataFrame))
    dict_of_data = attr.ib(validator=instance_of(dict))
    laser_freq = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    bidir = attr.ib(default=False, validator=instance_of(bool))
    tag_freq = attr.ib(default=189e3, validator=instance_of(float))
    tag_pulses = attr.ib(default=1, validator=instance_of(int))
    phase = attr.ib(default=-2.78, validator=instance_of(float))
    keep_unidir = attr.ib(default=False, validator=instance_of(bool))
    flim = attr.ib(default=False, validator=instance_of(bool))
    exp_params = attr.ib(factory=dict, validator=instance_of(dict))
    censor = attr.ib(default=False, validator=instance_of(bool))
    tag_interp_ok = attr.ib(default=False, validator=instance_of(bool))
    tag_to_phase = attr.ib(default=True, validator=instance_of(bool))
    tag_offset = attr.ib(default=0, validator=instance_of(int))
    deinterleave = attr.ib(default=False, validator=instance_of(bool))
    sorted_indices = attr.ib(init=False)
    bins_for_flim = attr.ib(init=False)

    def run(self):
        """ Pipeline of allocation """
        logging.info(
            "Channels of events found. Allocating photons to their frames and lines..."
        )
        self.__allocate_photons()
        self.__allocate_tag()
        if self.flim or self.deinterleave:
            self.df_photons, rel_time = self.__interpolate_laser(self.df_photons)
            if self.flim or self.censor:
                self._rotate_laser_timings()
        # Censor correction addition:
        if "Laser" not in self.dict_of_data.keys():
            self.dict_of_data["Laser"] = 0
        self.__reindex_dict_of_data()

        logging.info("Relative times calculated. Creating Movie object...")

    @property
    def num_of_channels(self):
        return sum([1 for key in self.dict_of_data.keys() if "PMT" in key])

    @property
    def hist_bins_between_laser_pulses(self):
        return int(np.ceil((1 / self.laser_freq) * 1e9))

    def __allocate_photons(self):
        """
        Returns a dataframe in which each photon has an address as its index - its corresponding
        frame, line, and possibly laser pulse. This address is used to histogram it in the right pixel.
        The arrival time of each photon is calculated relative to this address - start-of-line, for example.
        The TAG lens address is different - each photon simply receives a phase between 0 and 2π.

        The function doesn't return, but it populates self.df_photons with the allocated data.
        """
        irrelevant_keys = {"PMT1", "PMT2", "TAG Lens"}
        relevant_keys = set(self.dict_of_data.keys()) - irrelevant_keys
        # Preparations
        column_heads = {"Lines": "time_rel_line_pre_drop", "Laser": "time_rel_pulse"}

        # Main loop - Sort lines and frames for all photons and calculate relative time
        for key in reversed(sorted(relevant_keys)):
            self.dict_of_data[key].sort_values("abs_time", inplace=True)
            sorted_indices = (
                np.digitize(
                    self.df_photons.loc[:, "abs_time"].values,
                    self.dict_of_data[key].loc[:, "abs_time"].values,
                )
                - 1
            )
            self.df_photons[key] = (
                self.dict_of_data[key].iloc[sorted_indices, 0].values
            )  # columns 0 is abs_time,
            # but the .iloc method is amazingly faster than .loc
            positive_mask = sorted_indices >= 0
            # drop photons that came before the first line
            self.df_photons = self.df_photons.iloc[positive_mask].copy()
            # relative time of each photon in accordance to the line\frame\laser pulse
            if "Frames" != key:
                self.df_photons[column_heads[key]] = (
                    self.df_photons["abs_time"] - self.df_photons[key]
                )
            self.sorted_indices = sorted_indices[sorted_indices >= 0]
            if "Lines" == key:
                self.__rectify_photons_in_uneven_lines()

            self.df_photons.set_index(keys=key, inplace=True, append=True, drop=True)

        assert len(self.df_photons) > 0
        assert np.all(self.df_photons.iloc[:, 0].values >= 0)  # finds NaNs as well
        self.df_photons.sort_index(
            level=self.df_photons.index.names, axis=0, inplace=True, sort_remaining=True
        )
        self.df_photons.sort_index(
            level=self.df_photons.index.names, axis=1, inplace=True, sort_remaining=True
        )
        assert self.df_photons.index.is_lexsorted()

    def __allocate_tag(self):
        """ Allocate photons to TAG lens phase """
        try:
            tag = self.dict_of_data["TAG Lens"].loc[:, "abs_time"]
        except KeyError:
            return
        else:
            logging.info("Interpolating TAG lens data...")
            tag_pipe = TagPipeline(
                photons=self.df_photons,
                tag_pulses=tag,
                freq=self.tag_freq,
                binwidth=self.binwidth,
                num_of_pulses=self.tag_pulses,
                to_phase=self.tag_to_phase,
                offset=self.tag_offset,
            )
            tag_pipe.run()
            self.df_photons = tag_pipe.photons
            self.tag_interp_ok = tag_pipe.finished_pipe

            logging.info("TAG lens interpolation finished.")

    def __requires_censoring(self, data: np.ndarray) -> bool:
        """
        Method to determine if we should undergo the censor correction process

        :param np.ndarray data: Bins of histogram from their peak onward.
        :return bool:
        """
        diffs = np.diff(data)
        if len(diffs) == 0:
            return True
        if np.all(diffs <= 0):  # No censoring occurred
            return False
        else:
            first_idx = np.argwhere(diffs >= 0)[0][0]
            diffs2 = diffs[first_idx:]
            if np.all(diffs2 >= 0) or np.all(diffs2 <= 0):
                return True
            else:  # either false alarm, or a third photon is on its way
                self.__requires_censoring(diffs2)
                return True

    def __interpolate_laser(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Assign a time relative to a laser pulse for each photon. Assuming that the clock is synced to a 10 MHz signal.

        :param pd.DataFrame df: Dataframe with data for each photon.
        :return: Modified dataframe and the relative times.
        """
        needed_bins = self._find_integer_gcd()

        rel_time = []
        for chan in range(1, self.num_of_channels + 1):
            rel_time_first = (
                df.xs(key=chan, level="Channel", drop_level=False)["abs_time"].values
                % needed_bins
            )
            rel_time_per_pulse = rel_time_first % np.ceil(
                1 / (self.binwidth * self.laser_freq)
            ).astype(np.uint8)
            rel_time.append(rel_time_per_pulse)
            df.loc[chan, "time_rel_pulse"] = rel_time_per_pulse

        df["time_rel_pulse"] = np.uint8(df["time_rel_pulse"])
        return df, rel_time

    def _find_integer_gcd(self):
        """
        Converts the binwidth and laser frequency to the interger
        number of bins between pulses.
        """
        laser_freq = np.around(self.laser_freq, decimals=-6)  # MHz reprates
        bins_bet_pulses = np.around(laser_freq / self.binwidth, decimals=3)
        ns_between_pulses = (1 / laser_freq) * 1e9

        while int(bins_bet_pulses) != bins_bet_pulses:
            bins_bet_pulses *= 10

        while int(ns_between_pulses) != ns_between_pulses:
            ns_between_pulses *= 10

        return np.gcd(int(bins_bet_pulses), int(ns_between_pulses))

    def _find_num_pulses(self, needed_bins) -> int:
        period = int(needed_bins * self.binwidth * 1e9)
        laser_freq = np.around(self.laser_freq, decimals=-6)  # MHz reprates
        ns_between_pulses = (1 / laser_freq) * 1e9
        return int(period / ns_between_pulses)

    def __add_phase_offset_to_bidir_lines(self):
        """
        Uneven lines in a bidirectional scanning mode have to be offsetted.
        """
        phase_in_seconds = self.phase * 1e-6
        self.dict_of_data["Lines"].abs_time.iloc[1::2] -= np.uint64(
            phase_in_seconds / self.binwidth
        )

    def __rectify_photons_in_uneven_lines(self):
        """
        "Deal" with photons in uneven lines. Unidir - if keep_unidir is false, will throw them away.
        Bidir = flips them over (in the Volume object)
        """
        uneven_lines = np.remainder(self.sorted_indices, 2)
        if self.bidir:
            self.df_photons.rename(
                columns={"time_rel_line_pre_drop": "time_rel_line"}, inplace=True
            )

        elif not self.bidir and not self.keep_unidir:
            self.df_photons = self.df_photons.iloc[uneven_lines != 1, :].copy()
            self.df_photons.rename(
                columns={"time_rel_line_pre_drop": "time_rel_line"}, inplace=True
            )
            self.dict_of_data["Lines"] = (
                self.dict_of_data["Lines"].iloc[::2, :].copy().reset_index()
            )

        elif (
            not self.bidir and self.keep_unidir
        ):  # Unify the excess rows and photons in them into the previous row
            self.sorted_indices[np.logical_and(uneven_lines, 1)] -= 1
            self.df_photons.loc["Lines"] = (
                self.dict_of_data["Lines"].loc[self.sorted_indices, "abs_time"].values
            )
            self.dict_of_data["Lines"] = (
                self.dict_of_data["Lines"].iloc[::2, :].copy().reset_index()
            )
        try:
            self.df_photons.drop(["time_rel_line_pre_drop"], axis=1, inplace=True)
        except (ValueError, KeyError):  # column label doesn't exist
            pass
        self.df_photons = self.df_photons.loc[
            self.df_photons.loc[:, "time_rel_line"] >= 0, :
        ]

    def __reindex_dict_of_data(self):
        """
        Add new frame indices to the Series composing ``self.dict_of_data`` for slicing later on.
        The "Frames" indices are its values, and the "Lines" indices are the corresponding frames.
        """
        # Frames
        self.dict_of_data["Frames"] = pd.Series(
            self.dict_of_data["Frames"].abs_time.to_numpy(),
            index=self.dict_of_data["Frames"].abs_time.to_numpy(),
        )
        # Lines
        lines = self.dict_of_data["Lines"].abs_time.to_numpy()
        sorted_indices = np.digitize(lines, self.dict_of_data["Frames"].values) - 1
        positive_mask = sorted_indices >= 0
        lines = lines[positive_mask].copy()
        sorted_indices = sorted_indices[positive_mask]
        self.dict_of_data["Lines"] = pd.Series(
            lines, index=self.dict_of_data["Frames"].iloc[sorted_indices].values
        )

    def _rotate_laser_timings(self):
        """Rotate the histogram of the photon arrival times relative to the laser pulse,
        so that the highest value is in the second bin."""
        # bins_for_flim_hist = self._find_integer_gcd() + 1
        # hist, _ = np.histogram(self.df_photons["time_rel_pulse"], np.arange(bins_for_flim_hist))
        # peaks, params = find_peaks(hist, height=(None, None))
        # if len(peaks) == 0:
        #     return
        # max_peak_idx = (peaks[np.argmax(params["peak_heights"])] - 2) % bins_for_flim_hist
        # self.df_photons["time_rel_pules"] = (self.df_photons["time_rel_pulse"] - max_peak_idx) % bins_for_flim_hist
        pass
