from enum import Enum
from typing import NamedTuple
import warnings

import numpy as np
import attr
from attr.validators import instance_of

class TimepatchBits(NamedTuple):
    total: int
    time: int
    sweep: int
    tag: int
    lost: int


class Timepatch(Enum):
    Tp0 = TimepatchBits(16, 12, 0, 0, 0)
    Tp5 = TimepatchBits(32, 20, 8, 0, 0)
    Tp1 = TimepatchBits(32, 28, 0, 0, 0)
    Tp5b = TimepatchBits(64, 28, 16, 15, 1)
    TpDb = TimepatchBits(64, 28, 16, 16, 0)
    Tpf3 = TimepatchBits(64, 36, 7, 16, 1)
    Tp43 = TimepatchBits(64, 44, 0, 15, 1)
    Tpc3 = TimepatchBits(64, 44, 0, 16, 0)
    Tp3 = TimepatchBits(64, 54, 0, 6, 1)


@attr.s
class BinaryDataParser:
    """ Interpret data from binary list files """

    data = attr.ib(validator=instance_of(np.ndarray))
    timepatch = attr.ib(validator=instance_of(str))
    channel = attr.ib(init=False)
    edge = attr.ib(init=False)
    time = attr.ib(init=False)
    tag = attr.ib(init=False)
    sweep = attr.ib(init=False)
    lost = attr.ib(init=False)
    dict_of_data = attr.ib(init=False)
    timepatch_bits = attr.ib(init=False)

    def __attrs_post_init__(self):
        tpdict = {'0': Timepatch.Tp0,
                  '5': Timepatch.Tp5,
                  '1': Timepatch.Tp1,
                  '5b': Timepatch.Tp5b,
                  }
        try:
            self.timepatch_bits = tpdict[self.timepatch]
        except KeyError:
            raise KeyError(f"Invalid timepatch value received: {self.timepatch}.")

    def run(self):
        """ Main pipeline for the parsing """
        self.channel = self.__get_channel()
        self.edge = self.__get_edge()
        self.time = self.__get_time()
        if self.timepatch_bits.value.sweep != 0:
            self.sweep = self.__get_sweep()
        if self.timepatch != 'f3':
            if self.timepatch_bits.value.tag != 0:
                self.tag = self.__get_tag()
            if self.timepatch_bits.value.lost != 0:
                self.lost = self.__get_lost()
        else:
            self.tag = self.__get_tag_f3()
            self.lost = self.__get_lost_f3()


    def __get_channel(self) -> np.ndarray:
        """
        Parse the channel and edge information from the data.
        Return:
        -------
            :param chan np.ndarray: Array of channel numbers, with 6 == START
        """
        chan = (self.data & 0b111).astype(np.uint8)
        if np.any(chan > 6) or np.any(chan < 1):
            warnings.warn(f"Illegal channels found in file. Encountered the following"
                          f" values: {np.unique(chan)}.\nTrying to continue.")
        return chan

    def __get_edge(self) -> np.ndarray:
        """
        Parse the edge of each data line.
        Return:
        -------
            :param edge np.ndarray: Array of edges, 1 means falling edge.
        """
        return (np.right_shift(self.data, 3) & 1).astype(np.uint8)

    def __get_time(self) -> np.ndarray:
        """
        Parse the time bits.
        Return:
        -------
            :param time np.ndarray: Array of the absolute times for each event.
        """
        ones = int('1' * self.timepatch_bits.value.time, 2)  # time bits
        time = np.right_shift(self.data, 4) & ones
        return time.astype(np.uint64)

    def __get_sweep(self):
        """
        Parse the sweep bits.
        Return:
        -------
            :param sweep np.ndarray: Array of the sweep number for each event.
        """
        ones = int('1' * self.timepatch_bits.value.sweep, 2)
        right_shift_by = 4 + self.timepatch_bits.value.time
        sweep = np.right_shift(self.data, right_shift_by) & ones
        return sweep.astype(np.uint16)

    def __get_tag(self):
        """
        Parse the tag bits.
        Return:
        -------
            :param tag np.ndarray: Array of the tag bits for each event.
        """
        ones = int('1' * self.timepatch_bits.value.tag, 2)
        right_shift_by = 4 + self.timepatch_bits.value.time + \
            self.timepatch_bits.value.sweep
        tag = np.right_shift(self.data, right_shift_by) & ones
        return tag.astype(np.uint16)

    def __get_lost(self):
        """
        Parse the lost bits.
        Return:
        -------
            :param lost np.ndarray: Array of the lost bit for each event.
        """
        right_shift_by = self.timepatch_bits.value.total - 1
        lost = np.right_shift(self.data, right_shift_by) & 1
        return lost.astype(np.uint8)
