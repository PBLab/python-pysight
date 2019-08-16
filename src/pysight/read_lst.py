import attr
from attr.validators import instance_of
import numpy as np
import logging


@attr.s
class ReadData:
    """
    Read a given .lst file into memory

    :param str filename: File to read
    :param int start_of_data_pos: Number of header bytes in the file to skip
    :param str timepatch: Type of file
    :param bool is_binary: Whether the file is a binary or ASCII file
    :param bool debug: Run a debug build (limited number of lines for quick execution of the entire pipeline)
    """

    filename = attr.ib(validator=instance_of(str))
    start_of_data_pos = attr.ib(validator=instance_of(int))
    timepatch = attr.ib(validator=instance_of(str))
    is_binary = attr.ib(validator=instance_of(bool))
    debug = attr.ib(validator=instance_of(bool))
    data = attr.ib(init=False)

    def __attrs_post_init__(self):
        assert self.filename != ""
        assert self.start_of_data_pos > 0
        assert self.timepatch != ""

    @property
    def data_length(self):
        """  Number of bits in a word for each timepatch """
        return {
            "0": 16,
            "5": 32,
            "1": 32,
            "1a": 48,
            "2a": 48,
            "22": 48,
            "32": 48,
            "2": 48,
            "5b": 64,
            "Db": 64,
            "f3": 64,
            "43": 64,
            "c3": 64,
            "3": 64,
        }

    def read_lst(self) -> np.ndarray:
        """
        Main method to load the .lst file into memory.

        :return np.ndarray: An array with the ascii or binary data \
        that was read to memory using one of the helper functions.
        """
        bytes_to_add = self._check_carriage_return()
        data_length_bytes = self._get_data_length_bytes(self.timepatch, bytes_to_add)
        num_of_lines_in_bits = self._determine_num_of_lines(
            self.debug, data_length_bytes
        )
        # Read file
        if self.is_binary:
            self.data = self._read_binary(data_length_bytes, num_of_lines_in_bits)
        else:
            self.data = self._read_ascii(data_length_bytes, num_of_lines_in_bits)

        logging.info("File read. Sorting the file according to timepatch...")
        return self.data

    def _check_carriage_return(self) -> int:
        r""" If the file was generated in Windows each ASCII word will end with
        '\r\n' (2 bytes). Else it will probably end with '\n'.  """
        if self.is_binary:
            return 0
        with open(self.filename, "rb") as f:
            f.seek(self.start_of_data_pos)
            arr = np.fromfile(f, dtype="18S", count=18).astype("18U")
        first = arr[0]
        if "\r" in first:
            return 2
        else:
            return 1

    def _get_data_length_bytes(self, tp: str, bytes_to_add: int):
        r"""
        Number of bytes per word for each timepatch. For the hex
        case it takes into consideration the fact that each two letters
        correspond to a single byte.
        In addition, Windows-generated lst files end with '\r\n', while
        Mac/Linux ones might only have '\n', so we make sure to add these
        bytes in as well.
        """
        if self.is_binary:
            return self.data_length[tp] // 8
        else:
            return self.data_length[tp] // 4 + bytes_to_add  # 2 is \r\n, 1 is just \n

    def _determine_num_of_lines(self, debug, bytess):
        """ In debug runs we don't read all lines """
        if debug:
            num_of_lines = int(0.2e6 * bytess)  # 200k events is usually enough
            printstr = f'[DEBUG] Reading file "{self.filename}"...'
        else:
            num_of_lines = -1
            printstr = f'Reading file "{self.filename}"...'
        logging.info(printstr)
        return num_of_lines

    def _read_binary(self, data_length, num_of_lines) -> np.ndarray:
        with open(self.filename, "rb") as f:
            f.seek(self.start_of_data_pos)
            arr: np.ndarray = np.fromfile(
                f, dtype=f"u{data_length}", count=num_of_lines
            )
        return arr

    def _read_ascii(self, data_length, num_of_lines) -> np.ndarray:
        with open(self.filename, "rb") as f:
            f.seek(self.start_of_data_pos)
            arr = np.fromfile(
                f, dtype="{}S".format(data_length), count=num_of_lines
            ).astype("{}U".format(data_length))
        return arr
