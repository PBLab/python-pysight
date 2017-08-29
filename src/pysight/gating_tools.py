"""
__author__ = Hagai Hargil
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd


@attr.s(slots=True)
class GatedDetection(object):
    """ Gating the unneeded photons """

    raw = attr.ib(validator=instance_of(pd.DataFrame))
    reprate = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    data = attr.ib(init=False)

    @property
    def bins_bet_pulses(self) -> int:
        return int(np.ceil(1 / (self.reprate * self.binwidth)))

    @property
    def range_length(self) -> int:
        return int(self.bins_bet_pulses / 2)

    def run(self):
        """ Main pipeline of class """
        if 'time_rel_pulse' not in self.raw.columns:
            self.data = self.raw
            return
        self.__validate_time_rel_pulse()
        hist = self.__gen_hist()
        self.__discard_events(hist=hist)

    def __gen_hist(self) -> np.array:
        """ Create a histogram of the data """

        hist, _ = np.histogram(self.raw.time_rel_pulse, bins=self.bins_bet_pulses)
        return hist

    def __validate_time_rel_pulse(self):
        """ Discard events that aren't inside the right range of allowed times from a laser pulse """

        self.raw = self.raw.loc[self.raw.time_rel_pulse >= 0]
        self.raw = self.raw.loc[self.raw.time_rel_pulse < self.bins_bet_pulses]

    def __discard_events(self, hist: np.array):
        """ Throw away unneeded events """

        peak_idx = np.argmax(hist)
        lower_range = (peak_idx - 1) % self.bins_bet_pulses  # taking into consideration photons that came
                                                       # just before the peak, mainly due to resolution considerations
        upper_range = lower_range + self.range_length
        range_after_mod = np.arange(int(lower_range), int(upper_range)) % self.bins_bet_pulses
        mask = self.raw.time_rel_pulse.isin(range_after_mod)
        self.data = self.raw.loc[mask]
