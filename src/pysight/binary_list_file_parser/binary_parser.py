from enum import Enum
from typing import NamedTuple, Dict, Any
import logging
import numpy as np
import pandas as pd
import attr
from attr.validators import instance_of

from pysight.ascii_list_file_parser.file_io import ReadMeta


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
    Tp3 = TimepatchBits(64, 54, 0, 5, 1)


@attr.s
class BinaryDataParser:
    """ Interpret data from binary list files """

    data = attr.ib(validator=instance_of(np.ndarray))
    timepatch = attr.ib(validator=instance_of(str))
    data_range = attr.ib(default=0, validator=instance_of(int))
    bitshift = attr.ib(default=0, validator=instance_of(int))
    acq_delay = attr.ib(default=0, validator=instance_of(int))
    holdafter = attr.ib(default=0, validator=instance_of(int))
    dict_of_inputs = attr.ib(default=attr.Factory(dict), validator=instance_of(dict))
    use_tag_bits = attr.ib(default=False, validator=instance_of(bool))
    dict_of_inputs_bin = attr.ib(init=False)
    data_to_grab = attr.ib(init=False)
    channel = attr.ib(init=False)
    edge = attr.ib(init=False)
    time = attr.ib(init=False)
    tag = attr.ib(init=False)
    sweep = attr.ib(init=False)
    lost = attr.ib(init=False)
    dict_of_data = attr.ib(init=False)
    timepatch_bits = attr.ib(init=False)
    aligned_data = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.data_to_grab = ["abs_time"]
        tpdict = {
            "0": Timepatch.Tp0,
            "5": Timepatch.Tp5,
            "1": Timepatch.Tp1,
            "5b": Timepatch.Tp5b,
            "Db": Timepatch.TpDb,
            "f3": Timepatch.Tpf3,
            "43": Timepatch.Tp43,
            "c3": Timepatch.Tpc3,
            "3": Timepatch.Tp3,
        }
        try:
            self.timepatch_bits = tpdict[self.timepatch]
        except KeyError:
            raise KeyError(f"Invalid timepatch value received: {self.timepatch}.")

        self.dict_of_inputs_bin = {}
        for key, val in self.dict_of_inputs.items():
            self.dict_of_inputs_bin[key] = int(val, 2)

    def run(self):
        """ Main pipeline for the parsing """
        self.channel = self.__get_channel()
        self.__check_user_inputs()
        self.edge = self.__get_edge()
        self.time = self.__get_time()
        if self.timepatch_bits.value.sweep != 0:
            self.sweep = self.__get_sweep()
            # self.sweep = self.__unfold_sweeps(self.sweep)
        if self.timepatch != "f3":
            if self.timepatch_bits.value.tag != 0:
                self.tag = self.__get_tag()
            if self.timepatch_bits.value.lost != 0:
                self.lost = self.__get_lost()
        else:
            self.tag = self.__get_tag_f3()
            self.lost = self.__get_lost_f3()
        self.aligned_data = self.__gen_df()
        self.dict_of_data = self.__slice_df_to_dict()
        logging.info(
            "Sorted dataframe created. Starting to set the proper data channel distribution..."
        )

    def __get_channel(self) -> np.ndarray:
        """
        Parse the channel and edge information from the data.

        :return np.ndarray chan: Array of channel numbers, with 6 == START
        """
        chan = (self.data & 0b111).astype(np.uint8)
        if np.any(chan > 6):
            logging.warning(
                f"Illegal channels found in file. Encountered the following"
                f" values: {np.unique(chan)}.\nTrying to continue."
            )
        return chan

    def __get_edge(self) -> np.ndarray:
        """
        Parse the edge of each data line.

        :return np.ndarray edge: Array of edges, 1 means falling edge
        """
        return (np.right_shift(self.data, 3) & 1).astype(np.uint8)

    def __get_time(self) -> np.ndarray:
        """
        Parse the time bits.

        :return np.ndarray time: Array of the absolute times for each event.
        """
        ones = int("1" * self.timepatch_bits.value.time, 2)  # time bits
        time = np.right_shift(self.data, 4) & ones
        return time.astype(np.uint64)

    def __get_sweep(self) -> np.ndarray:
        """
        Parse the sweep bits.

        :return np.ndarray sweep: Array of the sweep number for each event.
        """
        ones = int("1" * self.timepatch_bits.value.sweep, 2)
        right_shift_by = 4 + self.timepatch_bits.value.time
        sweep = np.right_shift(self.data, right_shift_by) & ones
        return sweep.astype(np.uint16)

    def __unfold_sweeps(self, sweeps: np.ndarray) -> np.ndarray:
        """In some cases more than the real sweep number in the list file
        exceeds the available numbers of represented sweeps. That is, an
        experiment can be coducted using 1000 sweeps, while the sweep counter
        only has 8 bits. This causes 'folding' in the sweep values. The goal of
        this method is to unfold the sweep numbers such that they're (almost)
        monotonically increasing.
        """
        max_possible_sweep_value = (2 ** self.timepatch_bits.value.sweep) - 1
        if sweeps.max() < max_possible_sweep_value:
            return sweeps
        folding_idx = np.where(sweeps == max_possible_sweep_value)
        if len(folding_idx) == 1:
            return sweeps

    def __get_tag(self) -> np.ndarray:
        """
        Parse the tag bits.

        :return np.ndarray tag: Array of the tag bits for each event.
        """
        ones = int("1" * self.timepatch_bits.value.tag, 2)
        right_shift_by = (
            4 + self.timepatch_bits.value.time + self.timepatch_bits.value.sweep
        )
        tag = np.right_shift(self.data, right_shift_by) & ones
        return tag.astype(np.uint16)

    def __get_lost(self) -> np.ndarray:
        """
        Parse the lost bits.

        :return np.ndarray lost: Array of the lost bit for each event.
        """
        right_shift_by = self.timepatch_bits.value.total - 1
        lost = np.right_shift(self.data, right_shift_by) & 1
        return lost.astype(np.uint8)

    def __get_tag_f3(self) -> np.ndarray:
        """
        Parse the TAG bits of the f3 timepatch files.

        :return np.ndarray tag: Array of the TAG bits for each event.
        """
        tag = np.right_shift(self.data, 48) & 65535
        return tag.astype(np.uint16)

    def __get_lost_f3(self) -> np.ndarray:
        """
        Parse the lost bit of the f3 timepatch files.

        :return np.ndarray lost: Array of the lost bit for each event.
        """
        lost = np.right_shift(self.data, 47) & 1
        return lost.astype(np.uint8)

    def __check_user_inputs(self):
        """
        Assert that the channels that the user believe were recorded are actually there.
        Before sorting all photons make sure that no input is missing from the user. If it's missing
        the code will ignore this channel, but not raise an exception
        """

        actual_data_channels = set(np.unique(self.channel)).difference({0})
        if actual_data_channels != set(self.dict_of_inputs_bin.values()):
            logging.warning(
                "Channels that were inserted in GUI don't match actual data channels recorded. \n"
                f"The list files contains data in the following channels: {actual_data_channels}."
            )
            thrown_channels = 0
            keys_to_pop = []
            for key, item in self.dict_of_inputs_bin.items():
                if item not in actual_data_channels:
                    keys_to_pop.append(key)
                    thrown_channels += 1
            [self.dict_of_inputs_bin.pop(key) for key in keys_to_pop]

    def __gen_df(self) -> pd.DataFrame:
        """
        Align the acquired data into a single DataFrame

        :return pd.DataFrame:
        """
        df = pd.DataFrame(
            self.time, index=[self.channel, self.edge], columns=["abs_time"]
        )
        try:
            df.abs_time += (self.sweep - 1) * (
                (self.data_range * self.bitshift) + self.holdafter
            ) + (self.sweep * self.acq_delay)
        except AttributeError:
            pass
        try:
            tag_ser = pd.Series(self.tag, index=[self.channel, self.edge])
            df["tag"] = tag_ser
        except AttributeError:
            pass
        else:
            if self.use_tag_bits:
                self.data_to_grab.extend(["tag", "edge"])

        try:
            lost_ser = pd.Series(self.lost, index=[self.channel, self.edge])
            df["lost"] = lost_ser
        except AttributeError:
            pass

        df.index.names = ["analog_input", "edge"]
        return df

    def __slice_df_to_dict(self) -> dict:
        """
        Take the DataFrame of data and create a dictionary of data from it
        :return dict:
        """
        dict_of_data = {}
        for key, analog_chan in self.dict_of_inputs_bin.items():
            relevant_vals = self.aligned_data.xs(key=analog_chan, level=0).loc[
                :, self.data_to_grab
            ]
            if key in ["PMT1", "PMT2"]:
                dict_of_data[key] = relevant_vals.reset_index(drop=True)
                dict_of_data[key]["Channel"] = (
                    1 if "PMT1" == key else 2
                )  # channel is the spectral channel
            else:
                dict_of_data[key] = relevant_vals.sort_values(
                    by=["abs_time"]
                ).reset_index(drop=True)

        return dict_of_data


def binary_parsing(cur_file: ReadMeta, raw_data: np.ndarray, config: Dict[str, Any]):
    """Reads a binary file to memory."""
    binary_parser = BinaryDataParser(
        data=raw_data,
        data_range=cur_file.data_range,
        timepatch=cur_file.timepatch,
        bitshift=cur_file.bitshift,
        acq_delay=cur_file.acq_delay,
        holdafter=cur_file.time_after,
        use_tag_bits=config["tagbits"]["tag_bits"],
        dict_of_inputs=cur_file.dict_of_input_channels,
    )
    binary_parser.run()
    return binary_parser.data_to_grab, binary_parser.dict_of_data
