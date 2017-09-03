"""
__author__ = Hagai Hargil
"""
import attr
from attr.validators import instance_of
import pandas as pd
from pysight.apply_df_funcs import convert_bin_to_int
import numpy as np


@attr.s(slots=True)
class ParseTAGBits(object):
    """
    Take TAG bits and add their value to the photon dataframe
    """
    dict_of_data = attr.ib(validator=instance_of(dict))
    photons      = attr.ib(validator=instance_of(pd.DataFrame))
    bits_dict    = attr.ib(validator=instance_of(dict))
    use_tag_bits = attr.ib(default=False, validator=instance_of(bool))
    df_with_bits = attr.ib(init=False)

    @property
    def func_for_keyword(self):
        return {"Power": self.__parse_power,
                "Z axis": self.__parse_tag_lens,
                "Slow axis": self.__parse_slow_axis,
                "Fast axis": self.__parse_fast_axis}

    def gen_df(self):
        """
        Add the TAG bits as a column
        :return: pd.DataFrame: A new instance of the photon dataframe
        """
        if not self.use_tag_bits:
            self.photons.drop('tag', axis=1, inplace=True, errors='ignore')
            return self.photons

        assert 'tag' in self.dict_of_data['PMT1'].columns

        for key in self.bits_dict:
            self.func_for_keyword[key]()

        return self.photons

    def __parse_power(self):
        """
        Parse the bits of a power modulator
        :return:
        """
        proc_tag_bits = self.slice_string_arrays(self.dict_of_data['PMT1'].tag.values,
                                                 start=self.bits_dict.start,
                                                 end=self.bits_dict.end+1)

        int_tag_bits = convert_bin_to_int(proc_tag_bits)
        self.photons.loc[:, 'Power Factor'] = int_tag_bits

    def __parse_tag_lens(self):
        pass

    def __parse_slow_axis(self):
        pass

    def __parse_fast_axis(self):
        pass

    @staticmethod
    def slice_string_arrays(arr: np.array, start: int, end: int) -> np.array:
        """
        Slice an array of strings efficiently.
        Based on http://stackoverflow.com/questions/39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
        with modifications for Python 3.
        """
        b = arr.view('U1').reshape(len(arr), -1)[:, start:end]
        return np.fromstring(b.tostring(), dtype='U' + str(end - start))
