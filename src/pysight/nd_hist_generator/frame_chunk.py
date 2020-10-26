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

