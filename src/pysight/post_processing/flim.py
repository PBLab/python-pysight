"""
Methods to calculate the lifetime of the generated
stacks
"""
from typing import Union, Optional
import logging

import numpy as np
import scipy.optimize
import attr
from attr.validators import instance_of, in_

from pysight.nd_hist_generator.outputs import PySightOutput


def in_channel(instance, attribute, value):
    if value not in instance.data.available_channels:
        return ValueError


@attr.s
class LifetimeCalc:
    """
    Using a generated PySightOutput object, calculates the
    total lifetime in that stack
    """

    data = attr.ib(validator=instance_of(PySightOutput))
    channel = attr.ib(validator=[instance_of(int), in_channel])
    num_of_bins = attr.ib(default=125, validator=instance_of(int))
    raw_arrival_times = attr.ib(init=False)
    hist = attr.ib(init=False)
    params = attr.ib(init=False)
    cov = attr.ib(init=False)

    def fit_entire_fov(self) -> Optional["ExpFitParams"]:
        """
        Main method for class. Returns an ExpFitParams object
        which contains the result of the fitting done.
        """
        self.raw_arrival_times = self._get_photons()
        self.hist = self._gen_aligned_hist(self.raw_arrival_times)
        try:
            lowest_bin = self._get_low_bin_idx(self.hist)
        except ValueError:  # no valid lowest bin
            logging.warning("No valid decay curve found in the histogram.")
            return
        chopped_hist = self.hist[:lowest_bin]
        self.params, self.cov = self._fit_decay(chopped_hist)
        return ExpFitParams(self.params[0], self.params[1], self.params[2])

    def _get_photons(self):
        """
        Returns an array of the photon arrival times relative to the
        last laser pulse.
        """
        return self.data.photons.xs(
            self.channel, level="Channel"
        ).time_rel_pulse.to_numpy()

    def _gen_aligned_hist(self, photons):
        """ Creates a histogram from the photon list and then moves it so that the highest
        bin is in the start of it.
        """
        hist, bins = np.histogram(photons, bins=self.num_of_bins)
        hist = np.roll(hist, -hist.argmax())
        return hist

    def _get_low_bin_idx(self, hist):
        """
        Finds in the histogram the edge of the decay curve, i.e.
        the lowest bin.
        """
        lowest = hist.argmin()
        if lowest < 2:
            raise ValueError
        return lowest

    def _fit_decay(self, y):
        """
        Use a curve fitting method to find the parameters of the
        exponential decay of the data
        """
        xdata = np.arange(len(y)) / (self.data.config["binwidth"] * 10 ** 9)
        popt, pcov = scipy.optimize.curve_fit(
            self.single_exp, xdata, y, (y.max(), 4, np.median(y))
        )
        return popt, pcov

    @staticmethod
    def single_exp(x, amp, tau, c):
        """
        A single decaying exponential fit, to be used with
        SciPy's ``curve_fit`` function.

        Parameters:
        -----------
        :param np.ndarray x: "Time" vector
        :param Union[float, int] amp: Starting amplitude
        :param Union[float, int] tau: Expected lifetime
        :param Union[float, int] c: Baseline level of the function
        """
        return amp * np.exp(-x / tau) + c


@attr.s(frozen=True)
class ExpFitParams:
    """
    Returned parameters from fitting a decay curve to a
    single exponential decay function.

    Parameters:
        :param float amplitude: Initial amplitude of decay.
        :param float tau: Fluorophore lifetime.
        :param float c: Constant offset of the decay above baseline.
    """
    amplitude = attr.ib(validator=instance_of(float))
    tau = attr.ib(validator=instance_of(float))
    c = attr.ib(validator=instance_of(float))

