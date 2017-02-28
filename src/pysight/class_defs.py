"""
__author__ = Hagai Hargil
"""

import attr
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from numba import jit, float64, int32, int64


def validate_number_larger_than_zero(instance, attribute, value: int=0):
    """
    Validator for attrs module - makes sure line numbers and row numbers are larger than 0.
    """

    if value >= instance.attribute:
        raise ValueError("{} has to be larger than 0.".format(attribute))


@jit((float64[:](int64, int64, int32)), nopython=True, cache=True)
def create_linspace(start, stop, num):
    linspaces = np.linspace(start, stop, num)
    assert np.all(np.diff(linspaces) > 0)
    return linspaces


@attr.s
class Struct(object):
    """ Basic struct-like object for data keeping. """

    start = attr.ib()
    end = attr.ib()


@attr.s
class Movie(object):
    """
    A holder for Frame objects to be displayed consecutively.
    """

    data = attr.ib()
    reprate = attr.ib(validator=attr.validators.instance_of(float))
    num_of_cols = attr.ib(validator=attr.validators.instance_of(int))
    num_of_rows = attr.ib(validator=attr.validators.instance_of(int))
    name = attr.ib(validator=attr.validators.instance_of(str))
    binwidth = attr.ib(validator=attr.validators.instance_of(float))

    @property
    def list_of_frame_times(self) -> List:
        """ All frames start-times in the movie. """
        frame_times = np.unique(self.data.index.get_level_values('Frames'))
        if len(frame_times) > 1:
            diff_between_frames = np.mean(np.diff(frame_times))
        else:
            diff_between_frames = self.data['time_rel_frames'].max()

        frame_times = list(frame_times)
        frame_times.append(frame_times[-1] + diff_between_frames)
        return frame_times

    @property
    def deque_of_frames(self):
        """ Populate the deque containing the frames. """
        from collections import deque

        deque_of_frames = deque()
        list_of_frames = self.list_of_frame_times

        for idx, current_time in enumerate(list_of_frames[:-1]):  # populate deque with frames
            cur_data = self.data.xs(current_time, level='Frames')
            if not cur_data.empty:
                deque_of_frames.append(Frame(data=cur_data, num_of_lines=self.num_of_cols,
                                             num_of_rows=self.num_of_rows, number=idx,
                                             reprate=self.reprate, binwidth=self.binwidth, empty=False,
                                             end_time=(list_of_frames[idx + 1] - list_of_frames[idx])))
            else:
                deque_of_frames.append(Frame(data=cur_data, num_of_lines=self.num_of_cols,
                                             num_of_rows=self.num_of_rows, number=idx,
                                             reprate=self.reprate, binwidth=self.binwidth, empty=True,
                                             end_time=(list_of_frames[idx + 1] - list_of_frames[idx])))
        return deque_of_frames

    def create_tif(self):
        """ Create all frames, frame-by-frame, save them as tiff and return the stack. """

        from tifffile import TiffWriter
        from collections import namedtuple


        # Create a list containing the frames before showing them
        frames = []
        deque = self.deque_of_frames
        Frame = namedtuple('Frame', ('hist', 'x', 'y'))
        single_frame = Frame
        with TiffWriter('{}.tif'.format(self.name[:-4]), bigtiff=True) as tif:
            for idx, cur_frame in enumerate(deque):
                single_frame.hist, single_frame.x, single_frame.y = cur_frame.create_hist()
                frames.append(single_frame)
                tif.save(single_frame.hist)


@attr.s(slots=True)  # slots should speed up display
class Frame(object):
    """
    Contains all data and properties of a single frame of the resulting movie.
    """
    num_of_lines = attr.ib()
    num_of_rows = attr.ib()
    number = attr.ib(validator=attr.validators.instance_of(int))  # the frame's ordinal number
    data = attr.ib()
    reprate = attr.ib()  # laser repetition rate, relevant for FLIM
    end_time = attr.ib()
    binwidth = attr.ib()
    empty = attr.ib(default=False, validator=attr.validators.instance_of(bool))

    @property
    def __metadata(self) -> Dict:
        """
        Creates the metadata of the frames to be created, to be used for creating the actual images
        using histograms. Metadata can include the first photon arrival time, start and end of frames, etc.
        :return: Dictionary of all needed metadata.
        """

        metadata = {}
        jitter = 0.02  # 5% of jitter of the signals that creates frames

        # Frame metadata
        frame_start = 0
        frame_end = self.end_time
        metadata['Frame'] = Struct(start=frame_start, end=frame_end)

        # Lines metadata
        lines_start = 0
        unique_indices = np.unique(self.data.index.get_level_values('Lines'))
        diffs = np.diff(unique_indices)
        diffs_max = diffs.max()

        try:
            if diffs_max > ((1 + 4 * jitter) * np.mean(diffs)):  # Noisy data
                lines_end = np.mean(diffs) * (1 + jitter)
            else:
                lines_end = diffs_max
        except ValueError:
            lines_end = 0
            self.empty = True

        metadata['Lines'] = Struct(start=lines_start, end=lines_end)

        # Laser pulses metadata
        try:
            laser_start = 0
            laser_end =  1/self.reprate * self.binwidth  # 800 ps resolution
            metadata['Laser'] = Struct(start=laser_start, end=laser_end)
        except ValueError:
            pass

        return metadata

    def __create_hist_edges(self):
        """
        Create two vectors that will create the grid of the frame. Uses Numba internal function for optimization.
        :return: Tuple of np.array
        """
        metadata = self.__metadata

        if self.empty is not True:
            col_edge = create_linspace(start=metadata['Lines'].start,
                                       stop=metadata['Lines'].end,
                                       num=int(self.num_of_lines + 1))
            row_edge = create_linspace(start=metadata['Frame'].start,
                                       stop=metadata['Frame'].end,
                                       num=int(self.num_of_rows + 1))

            return col_edge, row_edge
        else:
            return 1, 1

    def create_hist(self):
        """
        Create the histogram of data using calculated edges.
        :return: np.ndarray of shape [num_of_cols, num_of_rows] with the histogram data, and edges
        """
        if not self.empty:
            xedges, yedges = self.__create_hist_edges()
            col_data_as_array = self.data["time_rel_line"].values
            row_data_as_array = self.data["time_rel_frames"].values

            hist, x, y = np.histogram2d(col_data_as_array, row_data_as_array, bins=(xedges, yedges))
        else:
            return np.zeros(self.num_of_lines, self.num_of_rows), 0, 0

        return hist, x, y

    def show(self):
        """ Show the frame. Mainly for debugging purposes, as the Movie object doesn't use it. """
        hist, x, y = self.create_hist()
        plt.figure()
        plt.imshow(hist, cmap='gray')
        plt.title('Frame number {}'.format(self.number))
        plt.axis('off')
