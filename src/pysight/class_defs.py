"""
__author__ = Hagai Hargil
"""

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from numba import jit, float64, uint64
from collections import OrderedDict
import warnings


def validate_number_larger_than_zero(instance, attribute, value: int=0):
    """
    Validator for attrs module - makes sure line numbers and row numbers are larger than 0.
    """

    if value >= instance.attribute:
        raise ValueError("{} has to be larger than 0.".format(attribute))


@jit((float64[:](uint64, uint64, uint64)), nopython=True, cache=True)
def create_linspace(start, stop, num):
    linspaces = np.linspace(start, stop, num)
    assert np.all(np.diff(linspaces) > 0)
    return linspaces


def metadata_ydata(data: pd.DataFrame, jitter: float=0.02):
    """
    Create the metadata for the y-axis.
    """
    lines_start = 0
    empty = False

    unique_indices = np.unique(data.index.get_level_values('Lines'))
    if unique_indices.shape[0] > 1:  # TODO: Doesn't make sense that there's a single line in a frame
        diffs = np.diff(unique_indices)
        diffs_max = diffs.max()
    else:
        diffs = unique_indices
        diffs_max = unique_indices

    try:
        if diffs_max > ((1 + 4 * jitter) * np.mean(diffs)):  # Noisy data
            lines_end = np.mean(diffs) * (1 + jitter)
        else:
            lines_end = int(diffs_max)
    except ValueError:
        lines_end = 0
        empty = True

    return lines_start, lines_end, empty


@attr.s
class Struct(object):
    """ Basic struct-like object for data keeping. """

    start = attr.ib()
    end = attr.ib()
    num = attr.ib(default=None)


@attr.s
class Movie(object):
    """
    A holder for Frame objects to be displayed consecutively.
    """

    data = attr.ib()
    reprate = attr.ib(validator=attr.validators.instance_of(float))
    x_pixels = attr.ib(validator=attr.validators.instance_of(int))
    y_pixels = attr.ib(validator=attr.validators.instance_of(int))
    z_pixels = attr.ib(validator=attr.validators.instance_of(int))
    name = attr.ib(validator=attr.validators.instance_of(str))
    binwidth = attr.ib(validator=attr.validators.instance_of(float))
    big_tiff = attr.ib(default=True)

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

    def choose_generator(self):
        """ Determine which generator to use - volumes or frames. """
        if 'Phase' in self.data.columns or 'time_rel_pulse' in self.data.columns:
            # self.big_tiff = False  # TODO: Not sure of the implication of BigTiffs on 4-5D data
            return self.gen_of_volumes()
        else:
            self.z_pixels = 1
            return self.gen_of_frames()

    def gen_of_frames(self):
        """ Populate the deque containing the frames as a generator. """

        list_of_frames = self.list_of_frame_times

        for idx, current_time in enumerate(list_of_frames[:-1]):  # populate deque with frames
            cur_data = self.data.xs(current_time, level='Frames')
            if not cur_data.empty:
                yield Frame(data=cur_data, num_of_lines=self.x_pixels,
                            num_of_rows=self.y_pixels, number=idx,
                            reprate=self.reprate, binwidth=self.binwidth, empty=False,
                            end_time=(list_of_frames[idx + 1] - list_of_frames[idx]))
            else:
                yield Frame(data=cur_data, num_of_lines=self.x_pixels,
                            num_of_rows=self.y_pixels, number=idx,
                            reprate=self.reprate, binwidth=self.binwidth, empty=True,
                            end_time=(list_of_frames[idx + 1] - list_of_frames[idx]))

    def gen_of_volumes(self):
        """ Populate the deque containing the volumes as a generator. """

        list_of_frames = self.list_of_frame_times
        for idx, current_time in enumerate(list_of_frames[:-1]):  # populate deque with frames
            cur_data = self.data.xs(current_time, level='Frames')
            if not cur_data.empty:
                yield Volume(data=cur_data, x_pixels=self.x_pixels,
                             y_pixels=self.y_pixels, z_pixels=self.z_pixels, number=idx,
                             reprate=self.reprate, binwidth=self.binwidth, empty=False,
                             end_time=(list_of_frames[idx + 1] - list_of_frames[idx]))
            else:
                yield Volume(data=cur_data, x_pixels=self.x_pixels,
                             y_pixels=self.y_pixels, z_pixels=self.z_pixels, number=idx,
                             reprate=self.reprate, binwidth=self.binwidth, empty=True,
                             end_time=(list_of_frames[idx + 1] - list_of_frames[idx]))

    def create_tif(self):
        """ Create all frames or volumes, one-by-one, and save them as tiff. """

        from tifffile import TiffWriter
        from collections import namedtuple

        # Create a list containing the frames before showing them
        relevant_generator = self.choose_generator()
        FrameTuple = namedtuple('Frame', ('hist', 'edges'))
        data_of_frame = FrameTuple

        try:
            cur_frame = next(relevant_generator)
        except StopIteration:
            raise ValueError('No frames generated.')

        with TiffWriter('{}.tif'.format(self.name[:-4]), bigtiff=self.big_tiff) as tif:
            while True:
                data_of_frame.hist, data_of_frame.edges = cur_frame.create_hist()
                tif.save(data_of_frame.hist.astype(np.uint16))
                try:
                    cur_frame = next(relevant_generator)
                except StopIteration:
                    break

    def create_array(self):
        """ Create all frames or volumes, one-by-one, and return the array of data that holds them. """
        from collections import namedtuple, deque

        # Create a deque and a namedtuple for the frames before showing them
        relevant_generator = self.choose_generator()
        FrameTuple = namedtuple('Frame', ('hist', 'edges'))
        data_of_frame = FrameTuple
        deque_of_frames = deque()

        try:
            cur_frame = next(relevant_generator)
        except StopIteration:
            raise ValueError('No frames generated.')

        while True:
            data_of_frame.hist, data_of_frame.edges = cur_frame.create_hist()
            deque_of_frames.append(data_of_frame)
            try:
                cur_frame = next(relevant_generator)
            except StopIteration:
                break

        assert len(deque_of_frames) == len(self.list_of_frame_times) - 1
        return deque_of_frames

    def create_single_volume(self, vols) -> np.array:
        """ Aggregate the frames\volumes that vols mark to a single frame\volume.
        :vols Volumes to aggregate. 'all' is all of them.
        :return np.array of histogram
        """

        deque_of_frames = self.create_array()

        if vols == 'all':
            summed = np.zeros_like(deque_of_frames[0].hist, dtype=np.int32)
            for frame in deque_of_frames:
                summed += frame.hist
            return summed

        if isinstance(vols, int):
            return deque_of_frames[vols]

        try:
            summed = np.array([0], dtype=np.int32)
            for idx in vols:
                summed += deque_of_frames[vols[idx]]
        except TypeError:
            raise TypeError('vols parameter should be "all", integer or list of volumes.')


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
    def __metadata(self) -> OrderedDict:
        """
        Creates the metadata of the frames to be created, to be used for creating the actual images
        using histograms. Metadata can include the first photon arrival time, start and end of frames, etc.
        :return: Dictionary of all needed metadata.
        """
        metadata = OrderedDict()
        jitter = 0.02  # 2% of jitter of the signals that creates frames

        # Frame metadata
        frame_start = 0
        frame_end = self.end_time
        metadata['Frame'] = Struct(start=frame_start, end=frame_end, num=self.num_of_rows + 1)

        # Lines metadata
        lines_start, lines_end, self.empty = metadata_ydata(data=self.data, jitter=jitter)
        metadata['Lines'] = Struct(start=lines_start, end=lines_end, num=self.num_of_lines + 1)

        return metadata

    def __create_hist_edges(self):
        """
        Create two vectors that will create the grid of the frame. Uses Numba internal function for optimization.
        :return: Tuple of np.array
        """
        metadata = self.__metadata
        list_of_edges = []

        if self.empty is not True:  # not an empty volume
            for key in metadata:
                list_of_edges.append(create_linspace(start=metadata[key].start,
                                                     stop=metadata[key].end,
                                                     num=metadata[key].num))

            return list_of_edges[0], list_of_edges[1]
        else:
            return 1, 1

    def create_hist(self):
        """
        Create the histogram of data using calculated edges.
        :return: np.ndarray of shape [num_of_cols, num_of_rows] with the histogram data, and edges
        """
        if not self.empty:
            xedges, yedges = self.__create_hist_edges()
            row_data_as_array = self.data["time_rel_line"].values
            col_data_as_array = self.data["time_rel_frames"].values

            hist, x, y = np.histogram2d(col_data_as_array, row_data_as_array, bins=(xedges, yedges))
        else:
            return np.zeros((self.num_of_lines, self.num_of_rows), dtype=np.int32), 0, 0

        return hist.astype(np.int32), (x, y)

    def show(self):
        """ Show the frame. Mainly for debugging purposes, as the Movie object doesn't use it. """
        hist, x, y = self.create_hist()
        plt.figure()
        plt.imshow(hist, cmap='gray')
        plt.title('Frame number {}'.format(self.number))
        plt.axis('off')


@attr.s(slots=True)
class Volume(object):
    """
    With a TAG lens, a movie is a sequence of volumes, rather than frames. Each volume contains frames in a plane.
    """
    # TODO: Refactor it with inheritance from Frame object in mind.

    x_pixels = attr.ib()
    y_pixels = attr.ib()
    z_pixels = attr.ib()
    number = attr.ib(validator=attr.validators.instance_of(int))  # the volume's ordinal number
    data = attr.ib()
    reprate = attr.ib()  # laser repetition rate, relevant for FLIM
    end_time = attr.ib()
    binwidth = attr.ib()
    empty = attr.ib(default=False, validator=attr.validators.instance_of(bool))

    @property
    def __metadata(self) -> OrderedDict:
        """
        Creates the metadata of the volume to be created, to be used for creating the actual images
        using histograms. Metadata can include the first photon arrival time, start and end of volume, etc.
        :return: Dictionary of all needed metadata.
        """

        metadata = OrderedDict()
        jitter = 0.02  # 2% of jitter of the signals that creates volumes

        # Volume metadata
        volume_start = 0
        volume_end = self.end_time
        metadata['Volume'] = Struct(start=volume_start, end=volume_end, num=self.x_pixels + 1)

        # y-axis metadata
        y_start, y_end, self.empty = metadata_ydata(data=self.data, jitter=jitter)
        metadata['Y'] = Struct(start=y_start, end=y_end, num=self.y_pixels + 1)

        # z-axis metadata
        if 'Phase' in self.data.columns:
            z_start = 0
            z_end = 2 * np.pi
            metadata['Z'] = Struct(start=z_start, end=z_end, num=self.z_pixels + 1)

        # Laser pulses metadata
        if 'time_rel_pulse' in self.data.columns:
            try:
                laser_start = 0
                laser_end = np.ceil(1 / (self.reprate * self.binwidth)).astype(int)
                metadata['Laser'] = Struct(start=laser_start, end=laser_end, num=laser_end + 1)
            except ZeroDivisionError:
                laser_start = 0
                warnings.warn('No laser reprate provided. Assuming 80 MHz.')
                laser_end = np.ceil(1 / (80e6 * self.binwidth)).astype(int)
                metadata['Laser'] = Struct(start=laser_start, end=laser_end, num=laser_end + 1)

        return metadata

    def __create_hist_edges(self):
        """
        Create three vectors that will create the grid of the frame. Uses Numba internal function for optimization.
        :return: Tuple of np.array
        """
        metadata = self.__metadata
        list_of_edges = []

        if self.empty is not True:
            for key in metadata:
                list_of_edges.append(create_linspace(start=metadata[key].start,
                                                     stop=metadata[key].end,
                                                     num=metadata[key].num))

            return list_of_edges
        else:
            return list(np.ones(len(metadata)))

    def create_hist(self):
        """
        Create the histogram of data using calculated edges.
        :return: np.ndarray of shape [num_of_cols, num_of_rows] with the histogram data, and edges
        """

        photon_identifiers = OrderedDict()

        if not self.empty:
            list_of_edges = self.__create_hist_edges()
            photon_identifiers['time_rel_frames'] = self.data['time_rel_frames'].values
            photon_identifiers['time_rel_line'] = self.data['time_rel_line'].values
            try:
                photon_identifiers['Phase'] = self.data['Phase'].values
            except KeyError:
                pass
            try:
                photon_identifiers['time_rel_pulse'] = self.data['time_rel_pulse'].values
            except KeyError:
                pass

            list_of_data_columns = [x for x in photon_identifiers.values()]

            data_to_be_hist = np.reshape(list_of_data_columns, (len(list_of_data_columns),
                                                                self.data.shape[0])).T

            assert data_to_be_hist.shape[0] == self.data.shape[0]
            assert len(photon_identifiers) == data_to_be_hist.shape[1]

            hist, edges = np.histogramdd(sample=data_to_be_hist, bins=list_of_edges)
        else:
            return np.zeros((self.x_pixels, self.y_pixels, self.z_pixels), dtype=np.int32), 0, 0, 0

        return hist.astype(np.int32), edges

    def show(self):
        """ Show the Volume. Mainly for debugging purposes, as the Movie object doesn't use it. """

        hist, edges = self.create_hist()
        plt.figure()
        # plt.imshow(hist, cmap='gray')
        plt.title('Volume number {}'.format(self.number))
        plt.axis('off')
