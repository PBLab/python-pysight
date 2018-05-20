import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


@attr.s(slots=True)
class FrameChunk:
    """
    Holds a chunk of data and can histogram ir efficiently.
    Composed out of a Movie - any attribute\method not definedhere is
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

        Returns:
        --------
        ``Dict[int, np.ndarray]`` A dictionary with its keys being the spectral channels
        of the data, and the value is a tuple of the histogrammed data and the edges.
        """
        self.hist_dict: Dict[int, Tuple[np.ndarray]] = {}
        for chan in self.df_dict:
            list_of_edges = self.__create_hist_edges(chan)
            data_columns = []
            data_columns.append(self.df_dict[chan]['abs_time'].values)
            data_columns.append(self.df_dict[chan]['time_rel_line'].values)
            try:
                data_columns.append(self.df_dict[chan]['Phase'].values)
            except KeyError:
                pass
            try:
                data_columns.append(self.df_dict[chan]['time_rel_pulse'].values)
            except KeyError:
                pass

            hist, edges = np.histogramdd(sample=data_columns, bins=list_of_edges)
            hist = hist.astype(np.uint8).reshape((self.frames_per_chunk,) + self.data_shape[1:])

            if self.bidir:
                hist[:, 1::2, ...] = np.flip(hist[:, 1::2, ...], axis=2)
            if self.censor:
                hist = self.__censor_correction(hist)

            self.hist_dict[chan] = hist, edges
        return self.hist_dict

    def __create_hist_edges(self, chan) -> List[np.ndarray]:
        """
        Generate the grid of the histogram.
        Inputs:
        -------

        :param chan: ``int`` Channel number

        Returns:
        --------

        ``list`` of ``np.ndarray``, one for each dimension
        """
        edges = []
        edges.append(self.__create_line_edges())
        edges.append(self.__create_col_edges())

        if 'Phase' in self.df_dict[chan].columns:
            edges.append(self.__linspace_along_sine())

        if 'time_rel_pulse' in self.df_dict[chan].columns:
            edges.append(self.__create_laser_edges())

        return edges

    def __censor_correction(self, hist):
        raise NotImplementedError("No censor correction as of yet. Contact package authors.")

    def __create_frame_edges(self) -> np.ndarray:
        """ Create edges for a numpy histogram for the frames dimension """

        assert self.frames.shape[0] == self.frames_per_chunk
        frames = np.hstack((self.frames.values, self.frames.values[-1] + np.uint64(1)))
        return frames

    def __create_line_edges(self) -> np.ndarray:
        """ Takes existing lines and turns them into bin edges. """

        assert self.lines.shape[0] <= self.x_pixels * self.frames_per_chunk  # last chunk can have less frames
        all_lines = np.hstack((self.lines.values, self.lines.values[-1] + self.line_delta))
        return all_lines

    def __create_col_edges(self) -> np.ndarray:
        if self.x_pixels == 1:
            return np.linspace(0, self.end_time, num=self.y_pixels+1, endpoint=True, dtype=np.uint64)

        delta = self.line_delta if self.bidir else self.line_delta / 2
        col_end = delta * self.fill_frac/100 if self.fill_frac > 0 else delta

        return np.linspace(start=0, stop=int(col_end), num=self.y_pixels+1, endpoint=True, dtype=np.uint64)

    def __linspace_along_sine(self) -> np.ndarray:
        """
        Find the points that are evenly spaced along a sine function between pi/2 and 3*pi/2
        :return: Array of bin edges
        """
        lower_bound = -1 if self.tag_as_phase else 0
        upper_bound = 1 if self.tag_as_phase else self.tag_period
        pts = []
        relevant_idx = []

        bin_edges = np.linspace(lower_bound, upper_bound, self.z_pixels+1, endpoint=True)[:, np.newaxis]
        dx = 0.00001
        x = np.arange(np.pi/2, 3*np.pi/2+dx, step=dx, dtype=np.float64)
        sinx = np.sin(x)
        locs = np.where(np.isclose(sinx, bin_edges, atol=1e-05))
        vals, first_idx, count = np.unique(locs[0], return_index=True, return_counts=True)
        assert len(vals) == len(bin_edges)
        for first_idx, count in zip(first_idx, count):
            idx_to_append = locs[1][first_idx + count // 2]
            relevant_idx.append(idx_to_append)
            pts.append(sinx[idx_to_append])

        return np.array(pts)
