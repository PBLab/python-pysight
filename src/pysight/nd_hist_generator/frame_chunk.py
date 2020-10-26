import functools
import operator
from typing import Dict, List, Tuple

import attr
import numpy as np
import pandas as pd
import modin.pandas as modin_pd
from attr.validators import instance_of

from pysight.nd_hist_generator.outputs import DataShape
from pysight.post_processing.flim import (
    calc_lifetime,
    flip_photons,
    add_downsample_frame_idx_to_df,
    add_bins_to_df,
)


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
            hist, _ = np.histogramdd(sample=data_columns, bins=list_of_edges)
            flim_hist = None
            if self.flim:
                list_of_edges_flim = self.__create_hist_edges(
                    chan, self.flim_downsampling_space
                )
                flim_hist = self._hist_with_flim(
                    self.df_dict[chan].copy(deep=True),
                    list_of_edges_flim,
                    list_of_edges,
                    chan,
                )
                flim_hist = flim_hist.reshape((-1, self.x_pixels, self.y_pixels,))
                num_of_frames = len(flim_hist)
                end_frame = num_of_frames if num_of_frames % 2 == 0 else num_of_frames - 1
                flim_hist = flim_hist[
                    slice(0, end_frame, self.flim_downsampling_time), :, :
                ]  # remove redundant frames
                division_factor = 1e-9 / self.binwidth  # results in nanoseconds
                flim_hist /= division_factor
            hists = self._post_process_hist([hist.astype(np.uint8)]) + (flim_hist,)
            self.hist_dict[chan] = hists


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

    def __create_hist_edges(self, chan, downsample=1) -> List[np.ndarray]:
        """
        Generate the grid of the histogram.

        :param int chan: Channel number
        :param int downsample: Downsampling factor

        :return List[np.ndarray]: One for each dimension
        """
        edges = []
        edges.append(self.__create_line_edges(downsample))
        edges.append(self.__create_col_edges(downsample))

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

    def __create_line_edges(self, downsample=1) -> np.ndarray:
        """ Takes existing lines and turns them into bin edges. """
        # last chunk can have less frames
        assert self.lines.shape[0] <= self.x_pixels * self.frames_per_chunk
        all_lines = np.hstack(
            (self.lines.to_numpy(), self.lines.to_numpy()[-1] + self.line_delta)
        )
        sampling_indices = np.linspace(
            0,
            len(self.lines),
            num=(len(self.lines) // downsample) + 1,
            endpoint=True,
            dtype=np.uint64,
        )
        return all_lines[sampling_indices]

    def __create_col_edges(self, downsample=1) -> np.ndarray:
        if self.x_pixels == 1:
            return np.linspace(
                0,
                self.end_time,
                num=(self.y_pixels // downsample) + 1,
                endpoint=True,
                dtype=np.uint64,
            )

        delta = self.line_delta if self.bidir else self.line_delta / 2
        start = 0
        if self.image_soft == "ScanImage":
            start = delta * 0.04
        col_end = delta * self.fill_frac / 100 if self.fill_frac > 0 else delta

        return np.linspace(
            start=start,
            stop=int(col_end),
            num=(self.y_pixels // downsample) + 1,
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
        self,
        data: pd.DataFrame,
        flim_edges: List[np.ndarray],
        edges: List[np.ndarray],
        chan: int,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        """Run a slightly more complex processing pipeline when we need to calculate
        the lifetime of each pixel in the image.

        Parameters
        ----------
        edges : list of np.ndarray
            Histogram edges for each dimension in the original image.
        flim_edges : list of np.ndarray
            Histogram edges for each dimension in the flim image.
        chan : int
            Channel number

        Returns
        -------
        hist_with_flim : np.ndarray
            the flim image, each pixel's value is a tau
        """
        if self.bidir:
            data = flip_photons(data, edges[0], self.lines, self.x_pixels)

        # add each photon bin in the time
        data = add_bins_to_df(data, flim_edges, ["abs_time", "time_rel_line"])

        data["downsampled_bin_of_dim0"] = data.bin_of_dim0 % (
            self.x_pixels // self.flim_downsampling_space
        )

        # downsample in time
        data = add_downsample_frame_idx_to_df(
            data, chan, self.frames, self.flim_downsampling_time
        )
        # estimate tau of group of photons
        data = modin_pd.DataFrame(data)
        data["tau"] = (
            data.loc[
                :,
                [
                    "time_rel_pulse",
                    "frame_idx",
                    "downsampled_bin_of_dim0",
                    "bin_of_dim1",
                ],
            ]
            .groupby(by=["frame_idx", "downsampled_bin_of_dim0", "bin_of_dim1"])
            .transform(calc_lifetime)
        )

        # create image from tau values
        data["lin_index"] = np.ravel_multi_index(
            [data["bin_of_dim0"], data["bin_of_dim1"]],
            [len(flim_edges[0]) - 1, len(flim_edges[1]) - 1],
        ).astype(int)
        flim_image = np.full(
            ((len(flim_edges[0]) - 1) * (len(flim_edges[1]) - 1),), np.nan
        )
        flim_image[data.loc[:, "lin_index"]] = data.loc[:, "tau"]
        flim_image = flim_image.reshape((-1, len(flim_edges[1]) - 1))

        # upsample image to original size
        bloater = np.ones(
            (self.flim_downsampling_space, self.flim_downsampling_space), dtype=np.uint8
        )
        return np.kron(flim_image, bloater)


@attr.s
class HistWithIndex:
    """A 'manual' implementation of np.histogramdd which also returns
    the indices of the partitioned photons, so that we could take these
    indices and copy them to be used with other data - the lifetime
    of these pixels, in our case.

    Parameters
    ----------
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
        ----------
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
    data_shape : tuple
        Dimension sizes (not including the time domain)
    downsample : int, optional
        How much downsampling should be conducted on the stack.
    """

    data = attr.ib(validator=instance_of(pd.Series))
    indices = attr.ib(validator=instance_of(pd.Series))
    mod_data_shape = attr.ib(validator=instance_of(tuple))
    downsample = attr.ib(default=16, validator=instance_of(int))
    bins_bet_pulses = attr.ib(default=125, validator=instance_of(int))
    all_data = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.all_data = pd.DataFrame({"since_laser": self.data, "bin": self.indices})
        self.original_data_shape = tuple(dim - 2 for dim in self.mod_data_shape)

    def partition_photons_into_bins(self):
        """Once we have the indices where each photon belongs, we can cluster them and
        calculate the lifetime of them all. This method clusters the photons into
        groups and sends this group off for lifetime calculation. The partitioning
        is dependent on the downsampling factor required by the user.
        """
        if len(self.mod_data_shape) > 1:
            assert self.mod_data_shape[0] == self.mod_data_shape[1]
        self.all_data["bin_per_frame"] = self.all_data["bin"] % (
            functools.reduce(operator.mul, self.mod_data_shape, 1)
        )
        blocks = self._create_block_matrix()
        self.all_data["block_num"] = blocks[self.all_data["bin_per_frame"]]
        hist_arrivals = self.all_data.groupby(
            "block_num", as_index=False, sort=False
        )  # .agg({"since_laser": calc_lifetime})
        self.all_data = self.all_data.set_index("block_num")
        self.all_data.loc[hist_arrivals["block_num"], "lifetime"] = hist_arrivals[
            "since_laser"
        ].astype(np.float32)

    def _create_block_matrix(self):
        """Generates a block matrix to be used as the downsampling window
        for FLIM calculation.

        When calculating the lifetime we downsample the data by binning nearby pixels.
        The block matrix returned from this function has an index corresponding
        to each block, but is the size of the original data. For example, an 8x8
        image downsampled by a factor of 2 will be expanded to 10x10, and the blocks
        will be the following:
        ``
        -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
        -1  1   1   2   2   3   3   4   4   -1
        -1  1   1   2   2   3   3   4   4   -1
        -1  5   5   6   6   7   7   8   8   -1
        -1  5   5   6   6   7   7   8   8   -1
        -1  9   9   10  10  11  11  12  12  -1
        -1  9   9   10  10  11  11  12  12  -1
        -1  13  13  14  14  15  15  16  16  -1
        -1  13  13  14  14  15  15  16  16  -1
        -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
        ``
        """
        ndim = len(self.mod_data_shape)
        number_of_blocks = self.original_data_shape[0] // self.downsample
        blocks = np.arange(number_of_blocks ** ndim, dtype=np.int32).reshape(
            ndim * (number_of_blocks,)
        )
        blocks = np.kron(blocks, np.ones(ndim * (self.downsample,), dtype=np.int8))
        blocks = np.pad(blocks, ndim * (1,), constant_values=-1)
        return blocks.ravel()

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
        shape = tuple(dim for dim in shape if dim != 1)
        total_bins = functools.reduce(operator.mul, shape, 1)
        assert total_bins >= self.all_data["bin_per_frame"].max()
        hist = np.full(shape, np.nan, dtype=np.float32).ravel()
        hist[self.all_data["bin_per_frame"]] = self.all_data["lifetime"]
        hist = hist.reshape(shape)
        core = len(shape) * (slice(1, -1),)
        # core = (slice(None),) + core  # when the time dimension is needed
        hist = hist[core]
        return hist


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
