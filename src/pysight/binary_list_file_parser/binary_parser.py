from enum import Enum
from typing import Tuple
import warnings

import numpy as np
import attr
from attr.validators import instance_of


class Timepatch(Enum):
    Tp0 = '0',
    Tp5 = '5',
    Tp1 = '1',
    # Tp5b =
    # TpDb =
    # Tpf3 =
    # Tp43 =
    # Tpc3 =
    # Tp3 =


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

    def run(self):
        """ Main pipeline for the parsing """
        self.channel = self.__get_channel()
        self.edge = self.__get_edge()


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


