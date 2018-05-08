import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
from typing import List, Dict


@attr.s(slots=True)
class FrameChunk:
    """
    Holds a chunk of data and can histogram ir efficiently.
    Composed out of a Movie - any attribute\method not definedhere is
    taken from the self.movie object with the __getattr__ method
    """
    movie = attr.ib()
    df_dict = attr.ib(validator=instance_of(dict), repr=False)
    frames_per_chunk = attr.ib(validator=instance_of(int))
    hist_dict = attr.ib(init=False)
    end_time = attr.ib(init=False)
    frames = attr.ib(init=False)

    def __getattr__(self, item):
        return getattr(self.movie, item)

    def __attrs_post_init__(self):
        self.frames = self.df_dict[1].index.get_level_values('Frames')
        frames_unique = np.unique(self.frames)
        if len(frames_unique) > 1:
            dif = np.diff(frames_unique).mean()
            self.end_time = frames_unique[-1] + dif
        else:
            self.end_time = self.frames[0] + 1

    def create_hist(self) -> Dict[int, np.ndarray]:
        """
        Main method to create the histogram of data. Assigns each event
        in the dataframe to its correct location, for each channel.
        :returns:
            :hist np.ndarray: n-dimensional histogram
            :edges tuple: arrays that represent edges
        """
        self.hist_dict: Dict[int, np.ndarray] = {}
        for chan in self.df_dict:
            list_of_edges = self.__create_hist_edges(chan)
            data_columns = []
            # data_columns.append(self.frames)
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
            # hist = np.vstack((hist, np.zeros(((1,) +  self.data_shape[2:]))))
            idx_to_take = np.ones_like(hist, dtype=np.bool)
            idx_to_take[np.arange(start=self.x_pixels, stop=self.x_pixels*self.frames_per_chunk, step=self.x_pixels+1), :] = False
            data = hist[idx_to_take].astype(np.uint8).reshape(((self.frames_per_chunk, ) + self.data_shape[1:]))

            if self.bidir:
                data[:, 1::2, ...] = np.fliplr(data[:, 1::2, ...])
            if self.censor:
                data = self.__censor_correction(data)

            self.hist_dict[chan] = data, edges
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
        edges.append(self.__create_frame_and_line_edges(chan))
        edges.append(self.__create_col_edges(chan))

        if 'Phase' in self.df_dict[chan].columns:
            edges.append(self.__linspace_along_sine())

        if 'time_rel_pulse' in self.df_dict[chan].columns:
            edges.append(self.__create_laser_edges())

        return edges

    def __censor_correction(self, hist):
        raise NotImplementedError("No censor correction as of yet. Contact package authors.")

    def __create_frame_and_line_edges(self, chan) -> np.ndarray:
        """
        Create edges for a numpy histogram for the frames and lines dimension. Its main job
        is to add a "closing" line for each frame, i.e. the right edge of the last bin of the image.
        If the lines originated from ScanImage, at the end of each frame there's a large difference between
        the last line of the previous frame and the first of the next frame. If lines are from MSCan,
        there's no such difference. This function has to deal with these two cases.
        """
        frames = np.unique(self.frames)
        assert frames.shape[0] == self.frames_per_chunk
        lines = np.unique(self.df_dict[chan].index.get_level_values('Lines'))
        assert len(lines) == self.x_pixels * self.frames_per_chunk

        frames_and_lines = lines.reshape((self.frames_per_chunk, self.x_pixels))
        mean_line_diffs = (np.diff(frames_and_lines, axis=1)).mean(axis=1, dtype=np.uint64)
        diff_bet_last_and_first = np.abs(frames_and_lines[1:, 0] - frames_and_lines[:-1, -1]).mean()
        if (diff_bet_last_and_first > self.line_delta * 1000) or (diff_bet_last_and_first < 3 * self.line_delta):
            # MSCan
            last_col_of_lines = np.atleast_2d(frames_and_lines[1:, 0] - 1).T  # -1 is due to bug in histogramdd
            last_edge = frames_and_lines[-1, -1] + mean_line_diffs.mean(dtype=np.uint64)
            last_col_of_lines = np.vstack((last_col_of_lines, last_edge))
            frames_and_lines = np.hstack((frames_and_lines, last_col_of_lines))
        else:  # ScanImage
            last_col_of_lines = np.atleast_2d(frames_and_lines[:, -1] + mean_line_diffs).T
            frames_and_lines = np.hstack((frames_and_lines, last_col_of_lines))
        return frames_and_lines.ravel()

    def __create_col_edges(self, chan) -> np.ndarray:
        num_of_lines = np.unique(self.df_dict[chan].index.get_level_values('Lines')).shape[0]
        if num_of_lines == 1:
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
