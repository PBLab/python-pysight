import attr
from attr.validators import instance_of
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable, Dict, Generator, Callable
from numba import jit, float64, uint64, int64
from collections import OrderedDict, namedtuple, deque
import warnings
import h5py_cache
from tqdm import tqdm

from pysight.nd_hist_generator.line_signal_validators.rectify_lines import LineRectifier
from .frame_chunk import FrameChunk


def trunc_end_of_file(name) -> str:
    """
    Take only the start of the filename to avoid error with Python and Windows
    :param name: File name to truncate.
    :return:
    """
    return name[:240]


@attr.s
class Movie(object):
    """
    A holder for Volume objects to be displayed consecutively.
    """
    data                = attr.ib(validator=instance_of(pd.DataFrame), repr=False)
    lines               = attr.ib(validator=instance_of(pd.Series), repr=False)
    frame_slices        = attr.ib(repr=False)  # generator of frame slices from VolumeGenerator
    frames              = attr.ib(validator=instance_of(pd.Series), repr=False)
    reprate             = attr.ib(default=80e6, validator=instance_of(float))
    name                = attr.ib(default='Movie', validator=instance_of(str),
                                  convert=trunc_end_of_file)
    binwidth            = attr.ib(default=800e-12, validator=instance_of(float))
    fill_frac           = attr.ib(default=71.0, validator=instance_of(float))
    bidir               = attr.ib(default=False, validator=instance_of(bool))
    num_of_channels     = attr.ib(default=1, validator=instance_of(int))
    outputs             = attr.ib(default={}, validator=instance_of(dict))
    censor              = attr.ib(default=False, validator=instance_of(bool))
    flim                = attr.ib(default=False, validator=instance_of(bool))
    lst_metadata        = attr.ib(default={}, validator=instance_of(dict))
    exp_params          = attr.ib(default={}, validator=instance_of(dict))
    line_delta          = attr.ib(default=158000, validator=instance_of(int))
    use_sweeps          = attr.ib(default=False, validator=instance_of(bool))
    cache_size          = attr.ib(default=10*1024**3, validator=instance_of(int))
    tag_as_phase        = attr.ib(default=True, validator=instance_of(bool))
    tag_freq            = attr.ib(default=189e3, validator=instance_of(float))
    mirror_phase        = attr.ib(default=-2.71, validator=instance_of(float))
    num_of_frame_chunks = attr.ib(default=1, validator=instance_of(int))
    frames_per_chunk    = attr.ib(default=1, validator=instance_of(int))
    data_shape          = attr.ib(default=(1, 512, 512), validator=instance_of(tuple))
    summed_mem          = attr.ib(init=False)
    stack               = attr.ib(init=False)
    x_pixels            = attr.ib(init=False)
    y_pixels            = attr.ib(init=False)
    z_pixels            = attr.ib(init=False)
    bins_bet_pulses     = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.x_pixels = self.data_shape[1]
        self.y_pixels = self.data_shape[2]
        self.z_pixels = 1
        self.bins_bet_pulses = 1
        try:
            self.z_pixels = self.data_shape[3]
            self.bins_bet_pulses = self.data_shape[4]
        except IndexError:
            pass

    @property
    def photons_per_pulse(self) -> Dict[int, float]:
        """ Caclculate the amount of detected photons per pulse """
        max_time = self.list_of_volume_times[-1] * self.binwidth
        num_of_pulses = int(max_time * self.reprate)
        photons_per_pulse = {}
        if self.num_of_channels == 1:
            photons_per_pulse[1] = self.data.shape[0] / num_of_pulses
        else:
            for chan in range(self.num_of_channels):
                photons_per_pulse[chan] = self.data.loc[chan]
        return photons_per_pulse

    def run(self) -> None:
        """
        Main pipeline for the movie object
        """
        funcs_during, funcs_end = self.__determine_outputs()
        self.__validate_df_indices()
        self.__process_data(funcs_during, funcs_end)
        self.__print_outputs()
        print("Movie object created, analysis done.")

    def __determine_outputs(self) -> Tuple[List[Callable], List[Callable]]:
        """
        Based on the "outputs" variable, decide which outputs to generate.
        Returns a list of pointers to functions to execute for every volume, and after
        finishing generating the entire stack.
        """
        if not self.outputs:
            warnings.warn("No outputs requested. Data is still accessible using the dataframe variable.")
            return [], []

        funcs_to_execute_during = []
        funcs_to_execute_end = []

        if 'memory' in self.outputs:
            self.summed_mem = {i: 0 for i in range(1, self.num_of_channels + 1)}
            self.stack = {i: deque() for i in range(1, self.num_of_channels + 1)}
            funcs_to_execute_during.append(self.__create_memory_output)
            funcs_to_execute_end.append(self.__convert_deque_to_arr)
            if 'stack' in self.outputs:
                funcs_to_execute_end.append(self.__save_stack_at_once)
            if 'summed' in self.outputs:
                funcs_to_execute_end.append(self.__save_summed_at_once)

        else:
            if 'stack' in self.outputs:
                self.outputs['stack'] = h5py_cache.File(f'{self.outputs["filename"]}', 'a',
                                                        chunk_cache_mem_size=self.cache_size,
                                                        libver='latest', w0=1).require_group('Full Stack')
                funcs_to_execute_during.append(self.__save_stack_incr)
                funcs_to_execute_end.append(self.__close_file)

            if 'summed' in self.outputs:
                self.summed_mem = {i: 0 for i in range(1, self.num_of_channels + 1)}
                funcs_to_execute_during.append(self.__append_summed_data)
                funcs_to_execute_end.append(self.__save_summed_at_once)

        return funcs_to_execute_during, funcs_to_execute_end


    def __process_data(self, funcs_during: List[Callable],
                         funcs_end: List[Callable]) -> None:
        """
        Create the outputs according to the outputs dictionary.
        Data is generated by appending to a list the needed micro-function to be executed.
        """
        # Execute the appended functions after generating each volume
        tq = tqdm(total=self.num_of_frame_chunks, desc=f"Processing frames...",
                  unit="frame", leave=False)
        for idx, frame_chunk in enumerate(self.frame_slices):
            sliced_df_dict, num_of_frames, frames, lines = self.__slice_df(frame_chunk)
            chunk = FrameChunk(movie=self, df_dict=sliced_df_dict, frames_per_chunk=num_of_frames,
                               frames=frames, lines=lines, )
            hist_dict = chunk.create_hist()
            for func in funcs_during:
                for chan, (hist, _) in hist_dict.items():
                    func(data=hist, channel=chan)

            tq.update(1)

        tq.close()
        [func() for func in funcs_end]

    def __slice_df(self, frame_chunk) -> Tuple[Dict[int, pd.DataFrame], int, pd.Series, pd.Series]:
        """
        Receives a slice object and slices the DataFrame accordingly -
        once per channel. The returned dictionary has a key for each channel.
        """
        slice_dict = {}
        idx_slice = pd.IndexSlice
        for chan in range(1, self.num_of_channels + 1):
            slice_dict[chan] = self.data.loc[idx_slice[chan, frame_chunk], :]
        frames = self.frames.loc[frame_chunk]
        num_of_frames = len(frames)
        lines = self.lines.loc[frame_chunk]
        if len(lines) > self.x_pixels * num_of_frames:
            warnings.warn(f"More-than-necessary line signals in the frame of chunk {frame_chunk}.")
        lines = lines.iloc[:self.x_pixels*num_of_frames]
        return slice_dict, num_of_frames, frames, lines

    def __validate_df_indices(self):
        """
        Make sure that the DataFrame of data contains the two
        important indices "Channel" and "Frames", and in the correct order.
        """
        if self.data.index.names[1] == 'Lines':
            self.data = self.data.swaplevel()

        assert self.data.index.names[0] == 'Channel'
        assert self.data.index.names[1] == 'Frames'
        assert self.data.index.names[2] == 'Lines'

    def __save_stack_at_once(self) -> None:
        """ Save the entire in-memory stack into .hdf5 file """
        with h5py_cache.File(f'{self.outputs["filename"]}', 'a', chunk_cache_mem_size=self.cache_size,
                             libver='latest', w0=1) as f:
            print("Saving full stack to disk...")
            for channel in range(1, self.num_of_channels + 1):
                f["Full Stack"][f"Channel {channel}"][...] = self.stack[channel]

    def __save_summed_at_once(self) -> None:
        """ Save the netire in-memory summed data into .hdf5 file """
        with h5py_cache.File(f'{self.outputs["filename"]}', 'a', chunk_cache_mem_size=self.cache_size,
                             libver='latest', w0=1) as f:
            for channel in range(1, self.num_of_channels + 1):
                f["Summed Stack"][f"Channel {channel}"][...] = np.squeeze(self.summed_mem[channel])

    def __close_file(self) -> None:
        """ Close the file pointer of the specific channel """
        self.outputs['stack'].file.close()

    def __convert_deque_to_arr(self) -> None:
        """ Convert a deque with a bunch of frames into a single numpy array with an extra
        dimension (0) containing the data.
        """
        for channel in range(1, self.num_of_channels + 1):
            self.stack[channel] = np.squeeze(np.vstack(self.stack[channel]))

    def __create_memory_output(self, data: np.ndarray, channel: int, **kwargs) -> None:
        """
        If the user desired, create two memory constructs -
        A summed array of all images (for a specific channel), and a stack containing
        all images in a serial manner.
        :param data: Data to be saved.
        :param channel: Current spectral channel of data
        """
        self.stack[channel].append(data)
        assert len(data.shape) > 2
        self.summed_mem[channel] += np.uint16(data.sum(axis=0))

    def __save_stack_incr(self, data: np.ndarray, channel: int) -> None:
        """
        Save incrementally new data to an open file on the disk
        :param data: Data to save
        :param channel: Current spectral channel of data
        :param vol_num: Current volume
        """
        self.outputs['stack'][f'Channel {channel}'][...] = np.squeeze(data)

    def __append_summed_data(self, data: np.ndarray, channel: int, **kwargs) -> None:
        """
        Create a summed variable later to be saved as the channel's data
        :param data: Data to be saved
        :param channel: Spectral channel of data to be saved
        """
        assert len(data.shape) > 2
        self.summed_mem[channel] += np.uint16(data.sum(axis=0))

    def __print_outputs(self) -> None:
        """
        Print to console the outputs that were generated.
        """
        if not self.outputs:
            return

        print('======================================================= \nOutputs:\n--------')
        if 'stack' in self.outputs:
            print(f'Stack file created with name "{self.outputs["filename"]}", \ncontaining a data group named'
                  ' "Full Stack", with one dataset per channel.')

        if 'memory' in self.outputs:
            print('The full data is present in dictionary form (key per channel) under `movie.stack`, '
                  'and in stacked form under `movie.summed_mem`.')

        if 'summed' in self.outputs:
            print(f'Summed stack file created with name "{self.outputs["filename"]}", \ncontaining a data group named'
                  ' "Summed Stack", with one dataset per channel.')

    def __nano_flim(self, data: np.ndarray) -> None:
        pass

    def show_summed(self, channel: int=1) -> None:
        """ Show the summed Movie """

        plt.figure()
        try:
            num_of_dims = len(self.summed_mem[channel].shape)
            if num_of_dims > 2:
                plt.imshow(np.sum(self.summed_mem[channel], axis=-(num_of_dims-2)),
                           cmap='gray')
            else:
                plt.imshow(self.summed_mem[channel], cmap='gray')
        except:
            warnings.warn("Can't show summed image when memory output wasn't asked for.")

        plt.title(f'Channel number {channel}')
        plt.axis('off')

    def show_stack(self, channel: int=1, slice_range: Iterable=range(10)) -> None:
        """ Show the stack of given slices """
        if 'time_rel_pulse' in self.data.columns:
            self.__show_stack_flim(channel=channel, slice_range=slice_range)
        else:
            self.__show_stack_no_flim(channel=channel, slice_range=slice_range)

    def __show_stack_no_flim(self, channel: int, slice_range: Iterable) -> None:
        """ Show the slices from the generated stack """

        img = None
        for frame in slice_range:
            print(frame, channel)
            if None == img:
                img = plt.imshow(self.stack[channel][frame, :, :], cmap='gray')
            else:
                img.set_data(self.stack[channel][frame, :, :])
            plt.pause(0.1)
            plt.draw()

    def __show_stack_flim(self, channel: int, slice_range: Iterable) -> None:
        """ Show the slices from the generated stack that contains FLIM data """

        for frame in slice_range:
            plt.figure()
            plt.imshow(np.sum(self.stack[channel][frame, :, :], axis=-1), cmap='gray')


@attr.s(slots=True)
class Volume(object):
    """
    A Movie() is a sequence of Volumes(). Each volume contains frames in a plane.
    """
    data           = attr.ib(validator=instance_of(pd.DataFrame))
    lines          = attr.ib(validator=instance_of(pd.Series))
    x_pixels       = attr.ib(default=512, validator=instance_of(int))
    y_pixels       = attr.ib(default=512, validator=instance_of(int))
    z_pixels       = attr.ib(default=1, validator=instance_of(int))
    number         = attr.ib(default=1, validator=instance_of(int))  # the volume's ordinal number
    reprate        = attr.ib(default=80e6, validator=instance_of(float))  # laser repetition rate, relevant for FLIM
    end_time       = attr.ib(default=np.uint64(100), validator=instance_of(np.uint64))
    binwidth       = attr.ib(default=800e-12, validator=instance_of(float))
    bidir          = attr.ib(default=False, validator=instance_of(bool))  # Bi-directional scanning
    fill_frac      = attr.ib(default=80.0, validator=instance_of(float))
    abs_start_time = attr.ib(default=np.uint64(0), validator=instance_of(np.uint64))
    empty          = attr.ib(default=False, validator=instance_of(bool))
    censor         = attr.ib(default=False, validator=instance_of(bool))
    line_delta     = attr.ib(default=158000, validator=instance_of(int))
    use_sweeps     = attr.ib(default=False, validator=instance_of(bool))
    tag_as_phase   = attr.ib(default=True, validator=instance_of(bool))
    tag_freq       = attr.ib(default=189e3, validator=instance_of(float))
    mirror_phase   = attr.ib(default=-2.76, validator=instance_of(float))  # phase for scanning mirrors


    @property
    def num_of_dims(self) -> int:
        """ Number of data dimensions """
        added_dims = 0
        if 'Phase' in self.data.columns:
            added_dims += 1
        if 'time_rel_pulse' in self.data.columns:
            added_dims += 1
        return 2 + added_dims

    @property
    def tag_period(self) -> int:
        return int(np.ceil(1 / (self.tag_freq * self.binwidth)))

    @property
    def dimensions_iterable(self) -> List:
        return [self.x_pixels, self.y_pixels, self.z_pixels, int(np.ceil(1 / (self.reprate * self.binwidth)))]

    def __create_hist_edges(self) -> Tuple[List, int]:
        """
        Create three vectors that will create the grid of the frame. Uses Numba internal function for optimization.
        :return: Tuple of np.array
        """
        list_of_edges = []
        if not self.empty:
            # Volume (row) edges
            try:
                list_of_edges.append(
                    LineRectifier(lines=self.lines.values - self.abs_start_time,
                                  x_pixels=self.x_pixels,
                                  bidir=self.bidir,
                                  end_time=self.end_time).rectify()
                )
            except ValueError:  # problem with line correction\interpolation
                warnings.warn(f"\nVolume {self.number} contained too many missing\corrupt lines.")
                list_of_edges.append(np.arange(self.x_pixels + 1))
            # Column edges
            y_start, y_end = metadata_ydata(data=self.data, jitter=0.02, bidir=self.bidir,
                                            fill_frac=self.fill_frac, delta=self.line_delta,
                                            sweeps=self.use_sweeps)
            list_of_edges.append(np.linspace(start=y_start,
                                             stop=self.end_time if y_end == 1 else y_end,
                                             num=self.y_pixels+1, endpoint=True))

            # Z edges
            if 'Phase' in self.data.columns:
                list_of_edges.append(self.__linspace_along_sine())

            # Laser pulses edges
            if 'time_rel_pulse' in self.data.columns:
                laser_start = 0
                try:
                    laser_end = np.ceil(1 / (self.reprate * self.binwidth)).astype(np.uint8)
                except ZeroDivisionError:
                    warnings.warn('No laser reprate provided. Assuming 80.3 MHz.')
                    laser_end = np.ceil(1 / (80.3e6 * self.binwidth)).astype(np.uint8)

                list_of_edges.append(np.linspace(start=laser_start,
                                                 stop=laser_end,
                                                 num=laser_end+1,
                                                 endpoint=True)[1:])

            return list_of_edges, self.num_of_dims
        else:
            return [], self.num_of_dims

    def create_hist(self) -> Tuple[np.ndarray, Iterable]:
        """
        Create the histogram of data using calculated edges.
        :return: np.ndarray of shape [num_of_cols, num_of_rows] with the histogram data, and edges
        """

        list_of_data_columns = []
        list_of_edges, num_of_dims = self.__create_hist_edges()
        if not self.empty:
            list_of_data_columns.append(self.data['time_rel_frames'].values)
            list_of_data_columns.append(self.data['time_rel_line'].values)
            try:
                list_of_data_columns.append(self.data['Phase'].values)
            except KeyError:
                pass
            try:
                list_of_data_columns.append(self.data['time_rel_pulse'].values)
            except KeyError:
                pass

            data_to_be_hist = np.reshape(list_of_data_columns, (num_of_dims, self.data.shape[0])).T

            assert data_to_be_hist.shape[0] == self.data.shape[0]
            assert len(list_of_data_columns) == data_to_be_hist.shape[1]

            hist, edges = np.histogramdd(sample=data_to_be_hist, bins=list_of_edges)
            if self.bidir:
                hist[1::2] = np.fliplr(hist[1::2])
            if self.censor:
                hist = self.__censor_correction(hist)

            return np.uint8(hist), edges
        else:
            return np.zeros(self.dimensions_iterable[:num_of_dims], dtype=np.uint8), (0, 0, 0)

    def __censor_correction(self, data) -> np.ndarray:
        """
        Add censor correction to the data after being histogrammed
        :param data:
        :return:
        """
        rel_idx = np.argwhere(np.sum(data, axis=-1) > 1)
        split = np.split(rel_idx, 2, axis=1)
        squeezed = np.squeeze(data[split[0], split[1], :])
        return data

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


def validate_number_larger_than_zero(instance, attribute, value: int=0):
    """
    Validator for attrs module - makes sure line numbers and row numbers are larger than 0.
    """

    if value >= instance.attribute:
        raise ValueError(f"{attribute} has to be larger than {value}.")


def metadata_ydata(data: pd.DataFrame, jitter: float=0.02, bidir: bool=True, fill_frac: float=0,
                   delta: int=158000, sweeps: bool=False) -> Tuple[int, int]:
    """
    Create the metadata for the y-axis.

    """
    lines_start: int = 0

    unique_indices: np.ndarray = np.unique(data.index.get_level_values('Lines'))
    if unique_indices.shape[0] <= 1:
        lines_end = 1
        return lines_start, lines_end

    # Case where it's a unidirectional scan and we dump back-phase photons
    if not bidir:
        delta /= 2

    if fill_frac > 0:
        lines_end = delta * fill_frac/100
    else:
        lines_end = delta

    return lines_start, int(lines_end)


@attr.s
class Struct(object):
    """ Basic struct-like object for data keeping. """

    start = attr.ib()
    end = attr.ib()
    num = attr.ib(default=None)
