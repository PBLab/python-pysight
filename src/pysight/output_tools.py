"""
__author__ = Hagai Hargil
"""

import attr
from attr.validators import instance_of
from pysight.movie_tools import trunc_end_of_file
import h5py
import numpy as np
import warnings
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
    outputs = attr.ib(init=False)

    def run(self):
        """ Parse what the user required, creating a list of HDF5 dataset pointers for each channel """
        self.outputs = {}
        if not self.output_dict:
            return
        self.outputs['memory'] = self.output_dict['memory']
        f = self.__create_prelim_file()
        if f is not None:
            data_shape_full = self.determine_data_shape_full()
            self.__populate_hdf(f, data_shape_full=data_shape_full)

    def __create_prelim_file(self):
        """ Try to create a preliminary .hdf5 file """
        if 'stack' in self.output_dict or 'summed' in self.output_dict:
            try:
                fullfile = f'{self.filename[:-4]}.hdf5'
                f = h5py.File(fullfile, 'w')
            except PermissionError or OSError:
                self.file_pointer_created = False
                warnings.warn("Permission Error: Couldn't write data to disk.")
                return
            return f
        else:
            return

    def __populate_hdf(self, f, data_shape_full):
        """
        Generate files and add metadata to each group
        f: File pointer
        """
        data_shape_summed = data_shape_full[:-1]
        if 'stack' in self.output_dict:
            try:
                self.outputs['stack'] = [f.require_group('Full Stack')
                                          .require_dataset(name=f'Channel {channel}',
                                                           shape=data_shape_full,
                                                           dtype=np.int16)
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
                                                            dtype=np.int16)
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
        """ Return the tuple that describes the shape of the final dataset """

        # Dimension order: [X, Y, Z, LIFETIME, FRAME]
        return np.squeeze(np.empty(shape=(self.x_pixels,
                                          self.y_pixels,
                                          self.z_pixels,
                                          self.bins_bet_pulses,
                                          self.num_of_frames),
                                   dtype=np.int8)).shape
