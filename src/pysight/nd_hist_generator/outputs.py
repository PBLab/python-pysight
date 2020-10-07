from typing import Dict, Any, Tuple, Union
from collections import namedtuple
import pathlib
import logging

import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import zarr
from numcodecs import Zstd


def trunc_end_of_file(name) -> str:
    """
    Take only the start of the filename to avoid error with Python and Windows

    :param str name: Filename to truncate
    :return str:
    """
    return name[:240]


@attr.s(slots=True)
class OutputParser:
    """
    Parse the wanted outputs and produce a dictionary with
    file pointers to the needed outputs
    # TODO: Handle the interleaved case
    """

    output_dict = attr.ib(validator=instance_of(dict))
    filename = attr.ib(validator=instance_of(str), converter=trunc_end_of_file)
    num_of_frames = attr.ib(default=1, validator=instance_of(int))
    flim_downsampling_time = attr.ib(default=1, validator=instance_of(int))
    x_pixels = attr.ib(default=512, validator=instance_of(int))
    y_pixels = attr.ib(default=512, validator=instance_of(int))
    z_pixels = attr.ib(default=1, validator=instance_of(int))
    channels = attr.ib(
        default=pd.CategoricalIndex([1]), validator=instance_of(pd.CategoricalIndex)
    )
    binwidth = attr.ib(default=100e-12, validator=instance_of(float))
    reprate = attr.ib(default=80e6, validator=instance_of(float))
    lst_metadata = attr.ib(factory=dict, validator=instance_of(dict))
    file_pointer_created = attr.ib(default=True, validator=instance_of(bool))
    cache_size = attr.ib(default=10 * 1024 ** 3, validator=instance_of(int))
    debug = attr.ib(default=False, validator=instance_of(bool))
    #: Dictionary with data - either the full or summed stack, with a list of zarr objects as channels
    outputs = attr.ib(init=False)
    #: Tuple of (num_of_frames, x, y, z, tau)
    data_shape = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)

    def run(self):
        """ Parse what the user required, creating a list of zarr dataset pointers for each channel """
        self.outputs = {}
        self.num_of_channels = len(self.channels)
        self.data_shape = self.determine_data_shape_full()
        if not self.output_dict:
            return
        if self.output_dict["memory"]:
            self.outputs["memory"] = 1
        self.__create_prelim_file()
        if self.file_pointer_created:
            self.__populate_file()

    @property
    def _group_names(self):
        return {
            "summed": "Summed Stack",
            "stack": "Full Stack",
            "flim": "Lifetime",
        }

    def __create_prelim_file(self):
        """ Try to create a preliminary zarr file. Cache improves IO performance """
        if (
            self.output_dict["stack"]
            or self.output_dict["summed"]
            or self.output_dict["flim"]
        ):
            try:
                path = pathlib.Path(self.filename)
                newpath = path.with_suffix(".zarr")
                debugged = "_DEBUG" if self.debug else ""
                fullfile = newpath.with_name(newpath.stem + debugged + newpath.suffix)
                self.outputs["filename"] = zarr.open(str(fullfile), mode="w")
                self.outputs["filename"].attr = self.lst_metadata
            except (PermissionError, OSError):
                self.file_pointer_created = False
                logging.warning("Permission Error: Couldn't write data to disk.")
            else:
                self.file_pointer_created = True
                self._generate_parent_groups()

    def _create_compressor(self):
        """Generate a compressor object for the Zarr array"""
        return Zstd(level=3)

    def _generate_parent_groups(self):
        """Create the summed stack / full stack groups in the zarr array."""
        for key, val in self.output_dict.items():
            if val is True:  # val can be a string
                if key != "memory":
                    self.outputs["filename"].require_group(self._group_names[key])
                self.outputs[key] = True

    def __populate_file(self):
        """
        Generate files and add metadata to each group, write out the data in chunks
        f: zarr file pointer
        """
        data_shape_summed = self.data_shape[1:]
        data_shape_flim = (
            int(np.ceil(self.data_shape[0] / self.flim_downsampling_time)),
        ) + self.data_shape[1:]
        chunk_shape = list(self.data_shape)
        chunk_shape[0] = 1
        chunk_shape = tuple(chunk_shape)
        if self.output_dict["stack"]:
            try:
                self.__create_group(
                    output_type="stack",
                    shape=self.data_shape,
                    chunks=chunk_shape,
                    dtype=np.uint8,
                )
            except (PermissionError, OSError):
                self.file_pointer_created = False
        if self.output_dict["summed"]:
            try:
                self.__create_group(
                    output_type="summed",
                    shape=data_shape_summed,
                    chunks=True,
                    dtype=np.uint16,
                )
            except (PermissionError, OSError):
                self.file_pointer_created = False

        if self.output_dict["flim"]:
            try:
                self.__create_group(
                    output_type="flim",
                    shape=data_shape_flim,
                    chunks=True,
                    dtype=np.float32,
                )
            except (PermissionError, OSError):
                self.file_pointer_created = False
        if self.file_pointer_created is False:
            logging.warning("Permission Error: Couldn't write data to disk.")

    def __create_group(
        self,
        output_type: str,
        shape: tuple,
        chunks: Union[Tuple, bool],
        dtype: np.dtype,
    ):
        """Create a group in the open file with the given parameters."""
        groupname = self._group_names[output_type]
        for channel in range(1, self.num_of_channels + 1):
            self.outputs["filename"][groupname].require_dataset(
                f"Channel {channel}",
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compressor=self._create_compressor(),
            )

    def determine_data_shape_full(self):
        """
        Return the tuple that describes the shape of the final dataset.
        Dimension order: [FRAME, X, Y, Z, LIFETIME]
        """
        non_squeezed = (
            self.x_pixels,
            self.y_pixels,
            self.z_pixels,
        )
        squeezed_shape = tuple([dim for dim in non_squeezed if dim != 1])
        return (
            self.num_of_frames,
        ) + squeezed_shape  # we never "squeeze" the number of frames


DataShape = namedtuple("DataShape", "t, x, y, z")


@attr.s(frozen=True)
class PySightOutput:
    """
    Keeps the relevant data from the run of the algorithm for later
    in-memory processing.

    :param pd.DataFrame photons: The 'raw' photon DataFrame.
    :param dict _summed_mem: Summed-over-time arrays of the data - one per channel.
    :param dict _stack: Full data arrays, one per channel.
    :param pd.CategoricalIndex _channels: Actual data channels analyzed.
    :param tuple _data_shape: Data dimensions
    :param bool _flim: Whether data has Tau channel.
    :param Dict[str,Any] config: Configuration file used in this run.
    """

    # TODO: Add new FLIM handler

    photons = attr.ib(validator=instance_of(pd.DataFrame), repr=False)
    _summed_mem = attr.ib(validator=instance_of(dict), repr=False)
    _stack = attr.ib(validator=instance_of(dict), repr=False)
    _channels = attr.ib(validator=instance_of(pd.CategoricalIndex), repr=False)
    _data_shape = attr.ib(validator=instance_of(tuple), repr=False)
    _flim = attr.ib(validator=instance_of(bool), repr=False)
    config = attr.ib(validator=instance_of(dict), repr=False)
    available_channels = attr.ib(init=False)
    data_shape = attr.ib(init=False)
    ch1 = attr.ib(init=False, repr=False)
    ch2 = attr.ib(init=False, repr=False)
    ch3 = attr.ib(init=False, repr=False)
    ch4 = attr.ib(init=False, repr=False)
    ch5 = attr.ib(init=False, repr=False)
    ch6 = attr.ib(init=False, repr=False)
    ch7 = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        """
        Populate the different attributes of the class
        """
        object.__setattr__(self, "available_channels", list(self._channels))
        object.__setattr__(self, "data_shape", self._parse_data_shape())
        for channel in self._channels:
            cur_stack = MultiDimensionalData(
                self._stack[channel], self._summed_mem[channel], self.data_shape
            )
            object.__setattr__(self, "ch" + str(channel), cur_stack)

    def _parse_data_shape(self):
        """
        Turns the data shape tuple into a namedtuple
        """
        shape = self._data_shape[:3]
        if len(self._data_shape) == 4:  # take TAG shape regardless
            shape += (self._data_shape[3],)
        else:
            shape += (None,)

        return DataShape(*shape)


# TODO: Add FLIM here
@attr.s(frozen=True)
class MultiDimensionalData:
    """
    Internal representation of a stack of data.

    :param np.ndarray full: The entirety of the data.
    :param np.ndarray time_summed: All data summed across the \
        time dimension.
    :param DataShape _data_shape: List of valid dimensions.
    :param np.ndarray z_summed: All data summed across the \
        z dimension.
    """

    full = attr.ib(validator=instance_of(np.ndarray), repr=False)
    time_summed = attr.ib(validator=instance_of(np.ndarray), repr=False)
    _data_shape = attr.ib(validator=instance_of(DataShape))
    z_summed = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        if self._data_shape.z:
            object.__setattr__(self, "z_summed", self.full.sum(axis=3))
