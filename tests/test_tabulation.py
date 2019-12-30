import numpy as np

from pysight.ascii_list_file_parser.tabulation import Tabulate, slice_string_arrays


class TestTabulation:

    tab = Tabulate(dict_of_inputs={}, data=np.array([]), dict_of_slices_hex={})

    def test_conversion_hex_to_bits(self):
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
        assert diction == self.tab.hex_to_bin_dict()

    def test_slice_string_array_1(self):
        a = np.array(["hello", "how", "are", "you"])
        assert np.array_equal(
            slice_string_arrays(a, 1, 3),
            np.array(["el", "ow", "re", "ou"], dtype="|U2"),
        )

    def test_slice_string_array_2(self):
        a = np.array(["hello", "how", "are", "you"])
        assert np.array_equal(
            slice_string_arrays(a, 0, 3),
            np.array(["hel", "how", "are", "you"], dtype="|U3"),
        )
