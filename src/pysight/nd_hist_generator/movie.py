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
    summed_mem          = attr.ib(init=False, repr=False)
    stack               = attr.ib(init=False, repr=False)
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
        tq = tqdm(total=self.num_of_frame_chunks, desc=f"Processing frame chunks...",
                  unit="chunk", leave=False)
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
        """ Save the entire in-memory summed data into .hdf5 file """
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


@attr.s
class Struct(object):
    """ Basic struct-like object for data keeping. """

    start = attr.ib()
    end = attr.ib()
    num = attr.ib(default=None)
