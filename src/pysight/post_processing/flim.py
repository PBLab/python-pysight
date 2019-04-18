"""
Methods to calculate the lifetime of the generated
stacks
"""
from typing import Union, Optional, Tuple, Iterator, Any
import logging
import itertools

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
    bin_factor = attr.ib(default=2, validator=instance_of(int))
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

    def flim_binned(self) -> np.ndarray:
        """
        Create an image of the FOV colored by the lifetime of each
        2x2 square in the image.
        """
        channel_data = getattr(self.data, f"ch{self.channel}")
        shape_x = channel_data.shape[0]
        shape_y = channel_data.shape[1]
        assert ( shape_x == shape_y)  # supports only square images ATM
        slices = self._gen_slices(shape_x)
        flim_img = self._populate_binned_image(slices, channel_data)
        return flim_img

    def _get_photons(self):
        """
        Returns an array of the photon arrival times relative to the
        last laser pulse.
        """
        return self.data.photons.xs(
            self.channel, level="Channel"
        ).time_rel_pulse.to_numpy()

    def _create_indices_per_ax(self, start, length=512) -> Tuple[Iterator[Any], ...]:
        """
        Create a vector which contains the indices at which we'll cut
        the original data in order to bin it.
        """
        idx = np.arange(start, length)
        idx = np.concatenate([None], idx, [None])
        idx2 = itertools.tee(idx)
        next(idx2[1])
        return idx2

    def _gen_slices(self, shape):
        """
        Create slice objects that will slice up the original
        array into a binned image.
        Each element of the returned list is a length 3 tuple, each
        of its elements being a slice, and each such slice details
        the starting and ending index of that bin.
        """

        idx_x = self._create_indices_per_ax(
            self.bin_factor, length=shape
        )
        idx_y = self._create_indices_per_ax(
            self.bin_factor, length=shape
        )
        xs = [slice(start, end) for start, end in zip(idx_x[0], idx_x[1])]
        ys = [slice(start, end) for start, end in zip(idx_y[0], idx_y[1])]
        return [(x, y, slice(None)) for x, y in zip(xs, ys)]

    def _populate_binned_image(self, slices, img):
        """
        Creates a binned image which contains the FLIM data from each of the
        original pixels of that bin.
        """
        flim_img = img.copy()
        for sl in slices:
            cur_data = img[sl].sum(axis=(0, 1))
            flim_img[sl] = cur_data
        return flim_img

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

