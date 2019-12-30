from typing import List, Dict, Tuple
import functools
import operator

import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pysight.nd_hist_generator.outputs import DataShape


@attr.s(slots=True)
class FrameChunk:
    """
    Holds a chunk of data and can histogram it efficiently.
    Composed out of a Movie - any attribute or method not defined here is
    taken from the self.movie object with the __getattr__ method
    """

    movie = attr.ib()
    df_dict = attr.ib(validator=instance_of(dict), repr=False)
    frames = attr.ib(validator=instance_of(pd.Series))
    lines = attr.ib(validator=instance_of(pd.Series))
    frames_per_chunk = attr.ib(validator=instance_of(int))
    hist_dict = attr.ib(init=False)
    end_time = attr.ib(init=False)

    def __getattr__(self, item):
        return getattr(self.movie, item)

    def __attrs_post_init__(self):
        if len(self.frames) > 1:
            dif = np.uint64(self.frames.diff().mean())
            self.end_time = self.frames.iloc[-1] + dif
        else:
            self.end_time = self.frames.iloc[0] + 1

    def create_hist(self) -> Dict[int, Tuple[np.ndarray]]:
        """
        Main method to create the histogram of data. Assigns each event
        in the dataframe to its correct location, for each channel.

        :return: A dictionary with its keys being the spectral channels
                of the data, and the value is a tuple of the histogrammed data and the edges.
        """
        self.hist_dict: Dict[int, Tuple[np.ndarray]] = {}
        for chan in self.df_dict:
            list_of_edges = self.__create_hist_edges(chan)
            data_columns: List[np.ndarray] = []
            data_columns.append(self.df_dict[chan]["abs_time"].to_numpy())
            data_columns.append(self.df_dict[chan]["time_rel_line"].to_numpy())
            try:
                data_columns.append(self.df_dict[chan]["Phase"].to_numpy())
            except KeyError:
                pass
            if self.flim:
                hist, flim_hist = self._hist_with_flim(
                    data_columns, list_of_edges, chan
                )
            else:
                flim_hist = None
                hist, _ = np.histogramdd(sample=data_columns, bins=list_of_edges)
            hists = self._post_process_hist([hist.astype(np.uint8)])
            # TODO: Throw this away once we do FLIM properly
            all_hists = hists + (flim_hist,)
            self.hist_dict[chan] = all_hists
        return self.hist_dict

    def _post_process_hist(self, hists: List[np.ndarray]):
        processed = []
        for hist in hists:
            if hist is not None:
                reshaped = hist.reshape(
                    (self.frames_per_chunk,) + (self.data_shape[1:])
                )
                if self.bidir:
                    reshaped[:, 1::2, ...] = np.flip(reshaped[:, 1::2, ...], axis=2)
                processed.append(reshaped)
            else:
                processed.append(None)
        return tuple(processed)

    def __create_hist_edges(self, chan) -> List[np.ndarray]:
        """
        Generate the grid of the histogram.

        :param int chan: Channel number

        :return ``list`` of ``np.ndarray``: One for each dimension
        """
        edges = []
        edges.append(self.__create_line_edges())
        edges.append(self.__create_col_edges())

        if "Phase" in self.df_dict[chan].columns:
            edges.append(self.__linspace_along_sine())

        return edges

    def __censor_correction(self, hist):
        raise NotImplementedError(
            "No censor correction as of yet. Contact package authors."
        )

    def __create_frame_edges(self) -> np.ndarray:
        """ Create edges for a numpy histogram for the frames dimension """

        assert self.frames.shape[0] == self.frames_per_chunk
        frames = np.hstack((self.frames.values, self.frames.values[-1] + np.uint64(1)))
        return frames

    def __create_line_edges(self) -> np.ndarray:
        """ Takes existing lines and turns them into bin edges. """

        assert (
            self.lines.shape[0] <= self.x_pixels * self.frames_per_chunk
        )  # last chunk can have less frames
        all_lines = np.hstack(
            (self.lines.values, self.lines.values[-1] + self.line_delta)
        )
        return all_lines

    def __create_col_edges(self) -> np.ndarray:
        if self.x_pixels == 1:
            return np.linspace(
                0, self.end_time, num=self.y_pixels + 1, endpoint=True, dtype=np.uint64
            )

        delta = self.line_delta if self.bidir else self.line_delta / 2
        start = 0
        if self.image_soft == "ScanImage":
            start = delta * 0.04
        col_end = delta * self.fill_frac / 100 if self.fill_frac > 0 else delta

        return np.linspace(
            start=start,
            stop=int(col_end),
            num=self.y_pixels + 1,
            endpoint=True,
            dtype=np.uint64,
        )

    def __linspace_along_sine(self) -> np.ndarray:
        """
        Find the points that are evenly spaced along a sine function between pi/2 and 3*pi/2

        :return: Array of bin edges
        """
        lower_bound = -1 if self.tag_as_phase else 0
        upper_bound = 1 if self.tag_as_phase else self.tag_period
        pts = []
        relevant_idx = []

        bin_edges = np.linspace(
            lower_bound, upper_bound, self.z_pixels + 1, endpoint=True
        )[:, np.newaxis]
        dx = 0.00001
        x = np.arange(np.pi / 2, 3 * np.pi / 2 + dx, step=dx, dtype=np.float64)
        sinx = np.sin(x)
        locs = np.where(np.isclose(sinx, bin_edges, atol=1e-05))
        vals, first_idx, count = np.unique(
            locs[0], return_index=True, return_counts=True
        )
        assert len(vals) == len(bin_edges)
        for first_idx, count in zip(first_idx, count):
            idx_to_append = locs[1][first_idx + count // 2]
            relevant_idx.append(idx_to_append)
            pts.append(sinx[idx_to_append])

        return np.array(pts)

    def _hist_with_flim(
        self, data: List[np.ndarray], edges: List[np.ndarray], chan: int
    ) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        """Run a slightly more complex processing pipeline when we need to calculate
        the lifetime of each pixel in the image.
        We first generate the standard histogram without taking the FLIM dimension
        into consideration, but we do keep the bin index of each of the photons
        in the image.
        Then we groupby the photons based on their hist index, and each such group
        goes to function which calculates the decay curve constants there.

        Parameters
        ----------
        data : list of np.ndarray
            Photon arrival times in each of the dimensions
        edges : list of np.ndarray
            Histogram edges for each dimension.
        chan : int
            Channel number

        Returns
        -------
        hist : np.ndarray
            N-dimensional histogram, where N = len(data)
        hist_with_flim : np.ndarray
            N-dimensional histogram, where N = len(data)
        """
        hist = HistWithIndex(data, edges)
        hist.run()
        valid_photons = hist.discard_out_of_bounds_photons()
        # TODO
        # flim = FlimCalc(self.df_dict[chan]["time_rel_pulse"].to_numpy(), hist.hist_indices)
        # flim.run()
        # only_flim_hist = np.full_like(hist.hist_photons, np.nan)
        # only_flim_hist[flim.hist_arrivals["bin"]] = flim.hist_arrivals["lifetime"]
        # assert only_flim_hist.shape == hist.hist_photons.shape
        return (
            hist.hist_photons,
            pd.DataFrame(
                {
                    "since_laser": self.df_dict[chan]["time_rel_pulse"].to_numpy()[
                        valid_photons
                    ],
                    "bin": hist.hist_indices[valid_photons],
                }
            ),
        )


@attr.s
class HistWithIndex:
    """A 'manual' implementation of np.histogramdd which also returns
    the indices of the partitioned photons, so that we could take these
    indices and copy them to be used with other data - the lifetime
    of these pixels, in our case.

    Parameters
    --------
    data : list of np.ndarray
        The data in a list, each dimension represnted as a vector of arrival times

    edges : list of np.ndarray
        The edges into which we should bin the photons
    """

    data = attr.ib(validator=instance_of(list))
    edges = attr.ib(validator=instance_of(list))
    hist_photons = attr.ib(init=False)
    hist_indices = attr.ib(init=False)
    edges_with_outliers = attr.ib(init=False)

    def __attrs_post_init__(self):
        """A few asserts regarding the data's shape and validity."""
        if len(self.data) != len(self.edges):
            raise TypeError(
                f"Data and bin length are unequal. Data length is {len(self.data)} while edges are {len(self.edges)}."
            )

    def run(self):
        """Main class pipeline."""
        self.hist_indices, self.edges_with_outliers = self._get_indices_for_photons()
        self.hist_photons = self._populate_hist_with_photons(self.edges_with_outliers)

    def _get_indices_for_photons(self):
        """For the given data and edges, find the indices in which each photon should
        belong. This is the first step in computing a histogram, and can be thought
        of as 'arghistogramdd'.
        """
        indices = tuple(
            np.searchsorted(edges, self.data[idx], side="right")
            for idx, edges in enumerate(self.edges)
        )

        nedges = []
        for dim, edge in enumerate(self.edges):
            on_edge = self.data[dim] == edge[-1]
            indices[dim][on_edge] -= 1
            nedges.append(len(edge) + 1)

        nedges = np.array(nedges)
        return np.ravel_multi_index(indices, nedges), nedges

    def _populate_hist_with_photons(self, nedges: np.ndarray):
        """Populates an empty n-d array with the photons in the indices
        which were calculate in "_get_indices_for_photons".

        Parameters
        --------

        nedges : np.ndarray
            number of edges per dimension

        """
        hist = np.bincount(self.hist_indices, minlength=nedges.prod())
        hist = hist.reshape(nedges)

        # remove outliers
        core = len(self.edges) * (slice(1, -1),)
        hist = hist[core]
        return hist

    def discard_out_of_bounds_photons(self):
        """The "run" methods automatically discards these photons, but if
        we wish to create a histogram by hand we have to discard them
        manually as well.
        This method has to be run after self._get_indices_for_photons has.
        """
        unraveled = np.unravel_index(self.hist_indices, self.edges_with_outliers)
        outliers = []
        for indices, edge in zip(unraveled, self.edges_with_outliers):
            outliers.append((indices != 0) & (indices < edge - 1))
        valid_photons = outliers[0]
        for indices in range(1, len(outliers)):
            valid_photons = np.logical_and(valid_photons, outliers[indices])
        return valid_photons


@attr.s(hash=True)
class FlimCalc:
    """An object designed to calculate the lifetime decay constant
    of the generated movie.
    It receives as input the photon arrival times and their binning index, and
    it then bins these arrival times and calculates the parameters of the
    observed exponential decay curve which rises from these bins. The final
    value of each of the bins is the tau calculated from that fit.

    Parameters
    --------
    data : np.ndarray
        The arrival times of all photons in the experiment

    indices : np.ndarray
        The bin indices of each of the photons

    downsample : int, optional
        How much downsampling should be conducted on the stack.
    """

    data = attr.ib(validator=instance_of(np.ndarray))
    indices = attr.ib(validator=instance_of(np.ndarray))
    downsample = attr.ib(default=10, validator=instance_of(int))
    bins_bet_pulses = attr.ib(default=125, validator=instance_of(int))
    all_data = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.all_data = pd.DataFrame({"since_laser": self.data, "bin": self.indices})

    def run(self):
        """Run the calculation pipeline."""
        self._partition_photons_into_bins()
        self._normalize_taus_to_uint8()

    def _partition_photons_into_bins(self):
        """Once we have the indices where each photon belongs, we can cluster them and
        calculate the lifetime of them all. This method clusters the photons into
        groups and sends this group off for lifetime calculation. The partitioning
        is dependent on the downsampling factor required by the user.
        """
        self.hist_arrivals = self.all_data.groupby(
            "bin", as_index=False, sort=False
        ).agg(calc_lifetime)

    def _normalize_taus(self):
        """FLIM images will be displayed in a float32 scale
        due to the nans.
        """
        self.hist_arrivals["lifetime"] = (self.hist_arrivals["since_laser"]).astype(
            np.float32
        )

    def histogram_result(self, shape: DataShape):
        """Create a histogram with the bins and lifetimes for each of the photons
        that were calculated in the "run" pipeline.

        Parameters
        ----------
        shape : DataShape
            Shape of data

        Returns
        -------
        hist : np.ndarray
            The histogrammed data
        """
        total_bins = functools.reduce(operator.mul, shape, 1)
        assert total_bins >= self.hist_arrivals["bin"]
        hist = np.full(shape, np.nan, dtype=np.float32).ravel()
        hist[self.hist_arrivals["bin"]] = self.hist_arrivals["lifetime"]
        hist = hist.reshape(shape)
        core = (len(shape) - 1) * (slice(1, -1),)
        core = (slice(None),) + core
        hist = hist[core]
        return hist


def calc_lifetime(data: pd.Series, bins_bet_pulses=124) -> float:
    """Calculate the lifetime of the given data by fitting it to a decaying exponent
    with a lifetime around 3 ns.
    # TODO: bins_bet_pulses
    """
    if len(data) < 5:
        return np.nan
    hist = np.histogram(data, bins_bet_pulses)[0]
    if hist.max() < 5:
        return np.nan
    peaks, props = find_peaks(hist, height=(None, None), prominence=(0.8, None))
    # If no peak is found we'll try using the first bin as a peak.
    # scipy.signal.find_peaks is not good at detecting that the first
    # data point is the highest. If it's not true then the curve fit
    # will eventually fail, leaving us with a nan instead of tau.
    if len(peaks) == 0:
        peaks, props = np.array([0]), {"peak_heights": hist[0]}
    decay_curve, max_val, min_val = find_decay_borders(hist, peaks, props)
    if len(decay_curve) < 4:
        return np.nan
    try:
        popt, _ = curve_fit(
            _exp_decay,
            np.arange(len(decay_curve)),
            decay_curve,
            p0=(max_val, 1 / 35, min_val),
            maxfev=1_000,
        )
    except RuntimeError:
        return np.nan
    tau = np.array(1 / popt[1]).astype(np.float32, casting="safe")
    if (tau > bins_bet_pulses) or (tau < 0):
        return np.nan
    return tau


def _exp_decay(x, a, b, c):
    """ Exponential function for FLIM and censor correction """
    return a * np.exp(-b * x) + c


def find_decay_borders(
    hist: np.ndarray, peaks: np.ndarray, props: Dict[str, np.ndarray]
):
    """Trims a given histogram which contains an exponential decay curve so that
    it starts at the peak and ends at the lowest point after it."""
    highest_peak_idx = peaks[np.argmax(props["peak_heights"])]
    decay_curve = hist[highest_peak_idx:]
    end_of_decay = np.argmin(decay_curve)
    decay_curve = decay_curve[: end_of_decay + 1]
    return decay_curve, decay_curve[0], decay_curve[-1]
