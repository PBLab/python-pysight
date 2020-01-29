import attr
from attr.validators import instance_of
import pandas as pd
import numpy as np
import itertools
from typing import Optional, Generator
import psutil


@attr.s(slots=True)
class VolumeGenerator:
    """
    Generate the list of volume chunks to be processed.
    Main method is "create_frame_slices", which returns a generator containing
    slice objects that signify the chunks of volumes to be processed simultaneously.

    :param pd.DataFrame frames: Frames for the entire dataset. Should not contain a closing, right-edge, frame.
    :param tuple data_shape: Shape of the final n-dimensional array (from the Output object)
    :param int MAX_BYTES_ALLOWED: Number of bytes that can be held in RAM. Calculated using the
                                  ``psutil`` package if not supplied manually.
    """

    frames = attr.ib(validator=instance_of(pd.Series), repr=False)
    data_shape = attr.ib(validator=instance_of(tuple))
    MAX_BYTES_ALLOWED = attr.ib(default=0, validator=instance_of(int))
    num_of_frames = attr.ib(init=False)
    bytes_per_frames = attr.ib(init=False)
    full_frame_chunks = attr.ib(init=False)
    frame_slices = attr.ib(init=False)
    frames_per_chunk = attr.ib(init=False)
    num_of_chunks = attr.ib(init=False)

    def __attrs_post_init__(self):
        if self.MAX_BYTES_ALLOWED == 0:
            try:
                avail = psutil.virtual_memory().available
            except AttributeError:
                self.MAX_BYTES_ALLOWED = 1_000_000_000
            else:
                self.MAX_BYTES_ALLOWED = avail // 32  # magic number

    def create_frame_slices(self, create_slices=True) -> Optional[Generator]:
        """
        Main method for the pipeline. Returns a generator with slices that
        signify the start time and end time of each chunk of frames. The indexing
        is inclusive-inclusive, and not inclusive-exclusive, since it's done using
        pandas' ``.loc`` method.

        :param bool create_slices: Used for testing, always keep ``True`` for actual code.
        """
        self.bytes_per_frames = np.prod(self.data_shape[1:]) * 8
        self.frames_per_chunk = int(
            min(
                max(1, self.MAX_BYTES_ALLOWED // self.bytes_per_frames),
                self.data_shape[0],
            )
        )
        self.num_of_frames = len(self.frames)
        self.num_of_chunks = int(max(1, len(self.frames) // self.frames_per_chunk))
        self.full_frame_chunks = self.__grouper()
        if create_slices:
            self.frame_slices = self.__generate_frame_slices()
            return self.frame_slices

    def __grouper(self) -> Generator:
        """ Chunk volume times into maximal-sized groups of values. """
        args = [iter(self.frames.to_numpy())] * self.frames_per_chunk
        return itertools.zip_longest(*args, fillvalue=np.nan)

    def __generate_frame_slices(self) -> Generator:
        if self.frames_per_chunk == 1:
            return (slice(frame[0], frame[0]) for frame in self.full_frame_chunks)

        start_and_end = []
        for chunk in self.full_frame_chunks:
            first, last = chunk[0], chunk[-1]
            if np.isnan(last):
                for val in reversed(chunk[:-1]):
                    if val is not np.nan:
                        last = val
                        break
            start_and_end.append((first, last))

        return (slice(np.uint64(t[0]), np.uint64(t[1])) for t in start_and_end)
