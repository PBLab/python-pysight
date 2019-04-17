"""
Methods to calculate the lifetime of the generated
stacks
"""
from typing import Union

import numpy as np
import scipy.optimize
import attr
from attr.validators import instance_of, in_

from pysight.nd_hist_generator.outputs import PySightOutput


@attr.s
class LifetimeCalc:
    """
    Using a generated PySightOutput object, calculates the
    total lifetime in that stack
    """

    data = attr.ib(validator=instance_of(PySightOutput))
    channel = attr.ib(validator=instance_of(int))
    raw_arrival_times = attr.ib(init=False)

    def __attrs_post_init__(self):
        assert self.channel in self.data.available_channels

    def fit(self):
        """
        Main method for class
        """
        self.raw_arrival_times = self._get_photons()

    def _get_photons(self):
        """
        Returns an array of the photon arrival times relative to the
        last laser pulse.
        """
        return self.data.photons.xs("Channel", self.channel).time_rel_pulse.to_numpy()

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

