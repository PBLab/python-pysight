"""
__author__ = Hagai Hargil
"""

import attr
from attr.validators import instance_of
from pysight.nd_hist_generator.movie import trunc_end_of_file
import numpy as np
import warnings
import h5py_cache
import os


@attr.s(slots=True)
class OutputParser(object):
    """
    Parse the wanted outputs and produce a dictionary with
    file pointers to the needed outputs
    """
    output_dict = attr.ib(validator=instance_of(dict))
    filename = attr.ib(validator=instance_of(str),
                       convert=trunc_end_of_file)
    num_of_frames = attr.ib(default=1, validator=instance_of(int))
    x_pixels = attr.ib(default=512, validator=instance_of(int))
    y_pixels = attr.ib(default=512, validator=instance_of(int))
    z_pixels = attr.ib(default=1, validator=instance_of(int))
    num_of_channels = attr.ib(default=1, validator=instance_of(int))
    flim = attr.ib(default=False, validator=instance_of(bool))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    reprate = attr.ib(default=80e6, validator=instance_of(float))
    lst_metadata = attr.ib(default={}, validator=instance_of(dict))
    file_pointer_created = attr.ib(default=True, validator=instance_of(bool))
    cache_size = attr.ib(default=10 * 1024**3, validator=instance_of(int))
    debug = attr.ib(default=False, validator=instance_of(bool))
    #: Dictionary with data - either the full or summed stack, with a list of HDF5 objects as channels
    outputs = attr.ib(init=False)
    #: Tuple of (num_of_frames, x, y, z, tau)
    data_shape = attr.ib(init=False)

    def run(self):
        """ Parse what the user required, creating a list of HDF5 dataset pointers for each channel """
        self.outputs = {}
        self.data_shape = self.determine_data_shape_full()
        if not self.output_dict:
            return
        try:
            self.outputs['memory'] = self.output_dict['memory']
        except KeyError:
            pass
        f = self.__create_prelim_file()
        if f is not None:
            self.__populate_hdf(f)

    def __create_prelim_file(self):
        """ Try to create a preliminary .hdf5 file. Cache improves IO performance """
        if 'stack' in self.output_dict or 'summed' in self.output_dict:
            try:
                split = os.path.splitext(self.filename)[0]
                debugged = '_DEBUG' if self.debug else ''
                fullfile = f'{split + debugged}.hdf5'
                f = h5py_cache.File(fullfile, 'w', chunk_cache_mem_size=self.cache_size, libver='latest', w0=1)
                self.outputs['filename'] = fullfile
            except PermissionError or OSError:
                self.file_pointer_created = False
                warnings.warn("Permission Error: Couldn't write data to disk.")
                return
            return f
        else:
            return

    def __populate_hdf(self, f):
        """
        Generate files and add metadata to each group, write out the data in chunks
        f: HDF5 file pointer
        """
        data_shape_summed = self.data_shape[1:]
        chunk_shape = list(self.data_shape)
        chunk_shape[0] = 1
        if 'stack' in self.output_dict:
            try:
                self.outputs['stack'] = [f.require_group('Full Stack')
                                          .require_dataset(name=f'Channel {channel}',
                                                           shape=self.data_shape,
                                                           dtype=np.uint8,
                                                           chunks=tuple(chunk_shape),
                                                           compression='gzip')
                                         for channel in range(1, self.num_of_channels + 1)]

                for key, val in self.lst_metadata.items():
                    for chan in range(self.num_of_channels):
                        self.outputs['stack'][chan].attrs.create(name=key, data=val.encode())

            except PermissionError or OSError:
                self.file_pointer_created = False
        if 'summed' in self.output_dict:
            try:
                self.outputs['summed'] = [f.require_group('Summed Stack')
                                           .require_dataset(name=f'Channel {channel}',
                                                            shape=data_shape_summed,
                                                            dtype=np.uint16,
                                                            chunks=True,
                                                            compression="gzip")
                                          for channel in range(1, self.num_of_channels + 1)]
                for key, val in self.lst_metadata.items():
                    for chan in range(self.num_of_channels):
                        self.outputs['summed'][chan].attrs.create(name=key, data=val.encode())

            except PermissionError or OSError:
                self.file_pointer_created = False

        f.close()
        if self.file_pointer_created is False:
            warnings.warn("Permission Error: Couldn't write data to disk.")

    @property
    def bins_bet_pulses(self) -> int:
        if self.flim:
            return int(np.ceil(1 / (self.reprate * self.binwidth)))
        else:
            return 1

    def determine_data_shape_full(self):
        """
        Return the tuple that describes the shape of the final dataset.
        Dimension order: [FRAME, X, Y, Z, LIFETIME]
        """
        non_squeezed = (self.x_pixels, self.y_pixels,
                        self.z_pixels, self.bins_bet_pulses)
        shape = tuple([dim for dim in non_squeezed if dim != 1])
        return (self.num_of_frames,) + shape  # we never "squeeze" the number of frames
