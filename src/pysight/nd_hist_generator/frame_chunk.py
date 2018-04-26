"""
__author__ = Hagai Har-Gil
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
from typing import List


@attr.s(slots=True)
class FrameChunk:
    """
    Holds a chunk of data and can histogram ir efficiently.
    Composed out of a Movie - any attribute\method not definedhere is
    taken from the self.movie object with the __getattr__ method
    """
    movie = attr.ib()
    df_dict = attr.ib(validator=instance_of(dict), repr=False)
    hist_dict = attr.ib(init=False)

    def __getattr__(self, item):
        return getattr(self.movie, item)

    def create_hist(self):
        """
        Main method to create the histogram of data. Assigns each event
        in the dataframe to its correct location, for each channel.
        :returns:
            :hist np.ndarray: n-dimensional histogram
            :edges tuple: arrays that represent edges
        """
        if not self.empty:
            for chan in self.df_dict:
                list_of_edges = self.__create_hist_edges(chan)
                data_columns = []
                data_columns.append(self.df_dict[chan].index.get_level_values('Frames'))
                data_columns.append(self.df_dict[chan]['time_rel_frames'].values)
                data_columns.append(self.df_dict[chan]['time_rel_line'].values)
                try:
                    data_columns.append(self.df_dict[chan]['Phase'].values)
                except KeyError:
                    pass
                try:
                    data_columns.append(self.df_dict[chan]['time_rel_pulse'].values)
                except KeyError:
                    pass

                data_to_be_hist = np.reshape(data_columns,
                                             (len(self.data_shape), self.df_dict[chan].shape[0])).T
                assert data_to_be_hist.shape[0] == self.df_dict[chan].shape[0]
                assert len(data_columns) == data_to_be_hist.shape[1]
                hist, edges = np.histogramdd(sample=data_to_be_hist, bins=list_of_edges)

                if self.bidir:
                    hist[1::2] = np.fliplr(hist[1::2])
                if self.censor:
                    hist = self.__censor_correction(hist)

                self.hist_dict[chan] = (np.uint8(hist), edges)
        else:
            self.hist_dict = {key: (np.zeros(self.data_shape, dtype=np.uint8),
                                    tuple([0] * len(self.data_shape)))
                              for key in self.df_dict}
        return self.hist_dict

    def __create_hist_edges(self, chan) -> List[np.ndarray]:
        """
        Generate the grid of the histogram.
        Inputs:
            :param chan int: Channel number
        Returns:
            list of np.ndarray - one for each dimension
        """
        edges = []
        edges.append(self.__create_frame_edges(chan))
        edges.append(self.__create_line_edges(chan))
        edges.append(self.__create_col_edges(chan))

        if 'Phase' in self.df_dict[chan].columns:
            edges.append(self.__linspace_along_sine())

        if 'time_rel_pulse' in self.df_dict[chan].columns:
            edges.append(self.__create_laser_edges())

        return edges

    def __censor_correction(self, hist):
        raise NotImplementedError("No censor correction as of yet. Contact package authors.")

    def __create_frame_edges(self, chan) -> np.ndarray:
        """ Create edges for a numpy histogram for the frames dimension """
        frames = np.unique(self.df_dict[chan].index.get_level_values('Frames'))
        print(frames)
        assert len(frames) == self.frames_per_chunk
        frames = np.hstack((frames, frames[-1] + np.uint64(1)))
        return frames

    def __create_line_edges(self, chan) -> np.ndarray:

        all_lines = np.unique(self.df_dict[chan].index.get_level_values('Lines'))
        assert len(all_lines) == self.x_pixels * self.frames_per_chunk
        # Add a closing line as the final edge
        all_lines = np.r_[all_lines, all_lines[-1] + np.uint64(np.diff(all_lines).mean())]
        return all_lines

    def __create_col_edges(self, chan) -> np.ndarray:
        num_of_lines = np.unique(self.df_dict[chan].index.get_level_values('Lines'))
        if num_of_lines == 1:
            return np.linspace(0, self.end_time, num=self.y_pixels+1, endpoint=True)

        delta = self.line_delta if self.bidir else self.linedelta / 2
        col_end = delta * self.fill_frac/100 if self.fill_frac > 0 else delta

        return np.linspace(start=0, stop=int(col_end), num=self.y_pixels+1, endpoint=True)

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
