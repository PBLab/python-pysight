from typing import Dict, List, Tuple
import logging
import re
from enum import Enum

from attr.validators import instance_of
import numpy as np
import attr


class LstFormat(Enum):
    ASCII = "ascii"
    BINARY = "binary"


@attr.s(slots=True)
class ReadMeta:
    """
    Manage pipeline of file IO process. Parses metadata, doesn't read the actual
    data in it.

    :param str filename: List file name
    :param str input_start: Data type in analog channel 'START' (6)
    :param str input_stop1: Data type in analog channel 'STOP1' (1)
    :param str input_stop2: Data type in analog channel 'STOP2' (2)
    :param float binwidth: Multiscaler resolution in seconds (100-800 picoseconds)
    :param bool use_sweeps: Use the sweeps counter as a new frame indicator (dev mode)
    """

    filename = attr.ib(validator=instance_of(str))
    debug = attr.ib(default=False, validator=instance_of(bool))
    input_start = attr.ib(default="Frames", validator=instance_of(str))
    input_stop1 = attr.ib(default="PMT1", validator=instance_of(str))
    input_stop2 = attr.ib(default="Lines", validator=instance_of(str))
    input_stop3 = attr.ib(default="Empty", validator=instance_of(str))
    input_stop4 = attr.ib(default="Empty", validator=instance_of(str))
    input_stop5 = attr.ib(default="Empty", validator=instance_of(str))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    use_sweeps = attr.ib(default=False, validator=instance_of(bool))
    mirror_phase = attr.ib(default=-2.78, validator=instance_of(float))
    is_binary = attr.ib(init=False)
    timepatch = attr.ib(init=False)
    data_range = attr.ib(init=False)
    time_after = attr.ib(init=False)
    acq_delay = attr.ib(init=False)
    start_of_data_pos = attr.ib(init=False)
    dict_of_input_channels = attr.ib(init=False)
    lst_metadata = attr.ib(init=False)
    fill_fraction = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)
    help_dict = attr.ib(init=False)

    def run(self):
        """ Open file and find the needed parameters """
        self.determine_binary()
        metadata: str = self.__get_metadata()
        self.timepatch: str = self.get_timepatch(metadata)
        if self.timepatch == "3" and not self.is_binary:
            raise NotImplementedError(
                'Timepatch value "3" is currently not supported for hex files. Please message the package owner.'
            )
        self.data_range: int = self.get_range(metadata)

        self.start_of_data_pos: int = self.get_start_pos()
        self.time_after: int = self.__get_hold_after(cur_str=metadata)
        self.acq_delay: int = self.__get_fstchan(cur_str=metadata)
        self.fill_fraction: float = self.__calc_actual_fill_fraction()
        self.dict_of_input_channels, self.num_of_channels = self.find_active_channels(
            metadata
        )
        # Grab some additional metadata from the file, to be saved later
        self.__parse_extra_metadata(metadata)

    def __attrs_post_init__(self):
        self.help_dict = {
            "1": (self.input_stop1, "001"),
            "2": (self.input_stop2, "010"),
            "3": (self.input_stop3, "011"),
            "4": (self.input_stop4, "100"),
            "5": (self.input_stop5, "101"),
            "6": (self.input_start, "110"),
        }

    def __get_metadata(self) -> str:
        """
        Read the file's metadata to be parsed by other functions.

        :return str: String with the .lst file's metadata
        """
        file_mode = "rb" if self.is_binary else "r"
        with open(self.filename, file_mode) as f:
            metadata = f.read(5000)

        return metadata

    @staticmethod
    def hex_to_bin_dict() -> Dict:
        """ Create a simple dictionary that maps a hex input into a 4 letter binary output. """
        diction = {
            "0": "0000",
            "1": "0001",
            "2": "0010",
            "3": "0011",
            "4": "0100",
            "5": "0101",
            "6": "0110",
            "7": "0111",
            "8": "1000",
            "9": "1001",
            "a": "1010",
            "b": "1011",
            "c": "1100",
            "d": "1101",
            "e": "1110",
            "f": "1111",
        }
        return diction

    def determine_binary(self):
        """
        Determine whether we're dealing with a binary or a hex file.
        """
        if self.filename == "":
            raise ValueError("No filename given.")

        try:
            with open(self.filename, "r") as f:
                txt = f.read(400)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.filename} doesn't exist.")
        except UnicodeDecodeError:
            with open(self.filename, "rb") as f:
                txt = f.read(400).decode()
        except:
            raise Exception(f"File {self.filename} read unsuccessfully.")

        reg = re.compile(r"\nmpafmt=(\w{3})")
        match = reg.findall(txt)
        if len(match) == 0:
            raise RuntimeError("Could not resolve file type.")
        if match[0] == "dat":
            self.is_binary = True
        elif match[0] == "asc":
            self.is_binary = False

    def get_range(self, cur_str) -> int:
        """
        Finds the "range" of the current file in the proper units
        :return int: range as defined by MCS6A
        """
        if self.filename == "":
            raise ValueError("No filename given.")

        if self.is_binary:
            format_str = rb"range=(\d+)"
        else:
            format_str = r"range=(\d+)"

        format_range = re.compile(format_str)
        range_lst = int(re.search(format_range, cur_str).group(1))

        return range_lst

    def get_timepatch(self, cur_str: str) -> str:
        """
        Get the time patch value out of of a list file.

        :param cur_str: Start of file to be analyzed.
        :return str: Time patch value as string.
        """
        if self.filename == "":
            raise ValueError("No filename given.")

        if self.is_binary:
            format_str: bytes = rb"time_patch=(\w+)"
        else:
            format_str: str = r"time_patch=(\w+)"

        format_timepatch = re.compile(format_str)
        timepatch = re.search(format_timepatch, cur_str).group(1)
        try:
            timepatch = timepatch.decode("utf-8")
        except AttributeError:
            assert isinstance(timepatch, str)
        finally:
            if self.is_binary and timepatch in ("1a", "2a", "22", "32", "2"):
                raise NotImplementedError(
                    f"The timepatch used ({timepatch}) isn't supported "
                    "for binary files since it uses a 6-byte word representation. "
                    "Please disallow this option in the MPANT software."
                )
            return timepatch

    def find_active_channels(self, cur_str) -> Tuple[Dict[str, str], int]:
        """
        Create a dictionary containing the active channels.

        :param cur_str: String to be analyzed.
        """
        if self.filename == "":
            raise ValueError("No filename given.")

        if self.is_binary:
            format_str = rb"\[CHN(\d)\].+?active=(\d)"
        else:
            format_str = r"\[CHN(\d)\].+?active=(\d)"

        format_active = re.compile(format_str, re.DOTALL)
        matches = format_active.findall(cur_str)
        dict_of_inputs = {}  # DataType: BinaryNumber (e.g. "Lines": "010")
        if self.is_binary:
            matches = [(match[0].decode(), match[1].decode()) for match in matches]
        for cur_match in matches[:-1]:
            if cur_match[1] == "1":  # channel is active, populate dict
                dict_of_inputs[self.help_dict[str(cur_match[0])][0]] = self.help_dict[
                    cur_match[0]
                ][1]
            elif (cur_match[1] == "0") and (
                self.help_dict[cur_match[0]][0] != "Empty"
            ):  # Inactive channel accroding to the multiscaler, but was marked as active by the user
                raise UserWarning(
                    f"Channel {cur_match[0]} didn't record data but was marked as active by the user."
                )

        if matches[-1][1] == "1":
            dict_of_inputs[self.help_dict["6"][0]] = self.help_dict["6"][1]
        elif (matches[-1][0] == "0") and (self.help_dict["6"][0] != "Empty"):
            raise UserWarning(
                "Channel 6 (START) didn't record data but was marked as active by the user."
            )

        num_of_channels = sum([1 for key in dict_of_inputs if "PMT" in key])

        assert len(dict_of_inputs) >= 1
        if "Empty" in dict_of_inputs.keys():
            logging.warning(
                f"At least one channel ({dict_of_inputs['Empty']})"
                " contained recorded data but was marked as 'Empty'."
            )
            dict_of_inputs.pop("Empty")

        return dict_of_inputs, num_of_channels

    def get_start_pos(self) -> int:
        """
        Returns the start position of the data

        :return int: Integer of file position for f.seek() method
        """
        if self.filename == "":
            raise ValueError("No filename given.")

        if self.is_binary:
            format_str = rb"DATA]\r\n"
            file_mode = "rb"
        else:
            format_str = r"DATA]\n"
            file_mode = "r"

        format_data = re.compile(format_str)
        pos_in_file: int = 0
        line_num: int = 0
        with open(self.filename, file_mode) as f:
            while pos_in_file == 0 and line_num < 1000:
                line = f.readline()
                line_num += 1
                match = re.search(format_data, line)
                if match is not None:
                    pos_in_file = f.tell()
                    return pos_in_file  # to have the [DATA] as header
        return -1

    def __get_hold_after(self, cur_str) -> int:
        """
        Read the time (in ns, and convert to timebins) that is considered a
        "hold after", or "hold-off", after a single sweep. Add that time, along with a 96 ns
        inherit delay, to all future times of the sweep.

        :param cur_str: String to parse
        :return int: Final time that has to be added to all sweeps, in timebins
        """
        if self.is_binary:
            format_str: bytes = rb"holdafter=([\w\+]+)"
        else:
            format_str: str = r"holdafter=([\w\+]+)"

        format_holdafter = re.compile(format_str)
        holdafter = float(re.search(format_holdafter, cur_str).group(1))
        EOS_DEADTIME = 96  # ns, current spec of Multiscaler

        time_after_sweep = int((EOS_DEADTIME + holdafter) * 10 ** (-9) / self.binwidth)
        return time_after_sweep

    def __get_fstchan(self, cur_str: str) -> int:
        """
        Read the acquisition delay of each sweep, called "fstchan" in the list files

        :param str cur_str: Metadata to be parsed
        :return int: Acq delay in timebins
        """
        if self.is_binary:
            format_str: bytes = rb"fstchan=(\w+)"
        else:
            format_str: str = r"fstchan=(\w+)"

        NANOSECONDS_PER_FSTCHAN = 6.4e-9  # Multiscaler const parameter

        format_fstchan = re.compile(format_str)
        fstchan = (
            int(re.search(format_fstchan, cur_str).group(1)) * NANOSECONDS_PER_FSTCHAN
        )  # in nanoseconds
        acq_delay = int(fstchan / self.binwidth)
        return acq_delay

    def __parse_extra_metadata(self, metadata):
        """
        Update self.lst_metadata with some additional information

        :param metadata: Data from the start of the lst file.
        """
        list_to_parse = [
            "fstchan",
            "holdafter",
            "periods",
            "rtpreset",
            "cycles",
            "sequences",
            "range",
            "sweepmode",
            "fdac",
        ]
        self.lst_metadata = {}
        for cur_str in list_to_parse:
            self.__parse_str(metadata, cur_str)

        self.lst_metadata["mirror_phase"] = str(self.mirror_phase)

    def __parse_str(self, metadata, str_to_parse):
        """
        Find str_to_parse in metadata and place the corresponding value in
        the dictionary self.lst_metadata
        """
        if self.is_binary:
            format_str: str = str_to_parse.encode("utf-8") + rb"=(\w+)"
        else:
            format_str: str = str_to_parse + r"=(\w+)"

        format_regex = re.compile(format_str)
        try:
            self.lst_metadata[str_to_parse] = str(
                re.search(format_regex, metadata).group(1)
            )
        except AttributeError:  # field is non-existent
            pass

    def __calc_actual_fill_fraction(self) -> float:
        """
        If we're using sweeps as lines then the true fill fraction is determined
        by the multiscaler's parameters, like hold_after and acquisiton delay.
        :return float: True fill fraction
        """
        if self.use_sweeps:
            fill_frac = self.data_range / (
                self.acq_delay + self.data_range + self.time_after
            )
            return fill_frac * 100  # in percent
        else:
            return -1.0
