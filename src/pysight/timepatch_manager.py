# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:59:09 2016

@author: Hagai
"""
import pandas as pd
from pysight.apply_df_funcs import b16, b22, hextobin
import numpy as np


class ChoiceManager:  # TODO: Single function with parameters...

    def __init__(self):
        self.__choice_table = \
        {
            "0": self.timepatch_0,
            "5": self.timepatch_5,
            "1": self.timepatch_1,
            "1a": self.timepatch_1a,
            "2a": self.timepatch_2a,
            "22": self.timepatch_22,
            "32": self.timepatch_32,
            "2": self.timepatch_2,
            "5b": self.timepatch_5b,
            "Db": self.timepatch_Db,
            "f3": self.timepatch_f3,
            "43": self.timepatch_43,
            "c3": self.timepatch_c3,
            "3": self.timepatch_3
        }

    def timepatch_0(self, data_range, df):
        df['abs_time'] = np.uint64(df['abs_time'].apply(b16, convert_dtype=False))

        df['sweep'] = 0
        df['tag'] = 0
        df['lost'] = 0
        return df


    def timepatch_5(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[2:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = df['raw'].str[0:2]
        df['sweep_as_str'] = df['sweep'].apply(b16, convert_dtype=False)
        df['abs_time'] += (df['sweep_as_str'] - 1) * data_range

        df['tag'] = 0
        df['lost'] = 0
        return df

    def timepatch_1(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[0:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = 0
        df['tag'] = 0
        df['lost'] = 0
        return df

    def timepatch_1a(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[4:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep_as_str'] = df['raw'].str[0:4]
        df['sweep'] = np.uint64(df['sweep_as_str'].apply(b16, convert_dtype=False))
        df['abs_time'] += (df['sweep'] - 1) * data_range

        df['tag'] = 0
        df['lost'] = 0
        return df

    def timepatch_2a(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[4:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep_as_str'] = df['raw'].str[2:4]
        df['sweep'] = df['sweep_as_str'].apply(b16, convert_dtype=False)
        df['abs_time'] += (df['sweep'] - 1) * data_range

        df['tag_as_str'] = df['raw'].str[0:2]
        df['tag'] = df['tag_as_str'].apply(b16, convert_dtype=False)

        df['lost'] = 0
        return df

    def timepatch_22(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[2:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = 0

        df['tag_as_str'] = df['raw'].str[0:2]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = 0
        return df

    def timepatch_32(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[2:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['bin'] = df['raw'].str[0:2].apply(hextobin)
        df['sweep_as_str'] = df['bin'].str[1:]
        df['sweep'] = np.uint64(df['sweep_as_str'].apply(b22, convert_dtype=False))
        df['abs_time'] += (df['sweep'] - 1) * data_range

        df['tag'] = 0

        df['lost'] = df['bin'].str[0].astype('category')

        return df

    def timepatch_2(self, data_range, df):

        df['abs_time_as_str'] = df['raw'].str[0:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = 0
        df['tag'] = 0
        df['lost'] = 0
        return df

    def timepatch_5b(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[8:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16), convert_dtype=False)

        df['sweep_as_str'] = df['raw'].str[4:8]
        df['sweep'] = np.uint64(df['sweep_as_str'].apply(b16, convert_dtype=False))
        df['abs_time'] += (df['sweep'] - 1) * data_range

        df['bin'] = df['raw'].str[0:4]
        df['bin'] = df['bin'].apply(hextobin)
        df['tag_as_str'] = df['bin'].str[3:18]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = df['bin'].str[2].astype('category')

        return df

    def timepatch_Db(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[8:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep_as_str'] = df['raw'].str[4:8]
        df['sweep'] = np.uint64(df['sweep_as_str'].apply(b16, convert_dtype=False))
        df['abs_time'] += (df['sweep'] - 1) * data_range

        df['tag_as_str'] = df['raw'].str[0:4]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = 0
        return df

    def timepatch_f3(self, data_range, df):

        df['abs_time_as_str'] = df['raw'].str[6:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['bin'] = df['raw'].str[4:6]
        df['bin'] = df['bin'].apply(hextobin)
        df['sweep_as_str'] = df['bin'].str[3:10]
        df['sweep'] = np.uint64(df['sweep_as_str'].apply(b16, convert_dtype=False))
        df['abs_time'] += (df['sweep'] - 1) * data_range

        df['tag_as_str'] = df['raw'].str[0:4]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = df['bin'].str[2].astype('category')

        df.drop(['bin'], axis=1, inplace=True)
        return df

    def timepatch_43(self, data_range, df):

        df['abs_time_as_str'] = df['raw'].str[4:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = 0

        df['bin'] = df['raw'].str[0:4]
        df['bin'] = df['bin'].apply(hextobin)
        df['tag_as_str'] = df['bin'].str[3:10]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = df['bin'].str[2].astype('category')
        return df

    def timepatch_c3(self, data_range, df):
        df['abs_time_as_str'] = df['raw'].str[4:-1]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = 0

        df['tag_as_str'] = df['raw'].str[0:4]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = 0
        return df

    def timepatch_3(self, data_range, df):
        df['bin'] = df['raw'].str[0:-1]
        df['bin'] = df['raw'].apply(hextobin)

        df['abs_time_as_str'] = df['bin'].str[8:62]
        df['abs_time'] = np.uint64(df['abs_time_as_str'].apply(b16, convert_dtype=False))

        df['sweep'] = 0

        df['tag_as_str'] = df['bin'].str[3:8]
        df['tag'] = np.uint64(df['tag_as_str'].apply(b16, convert_dtype=False))

        df['lost'] = df['bin'].str[2].astype('category')

        return df

    def process(self, case, data_range, df):
        """
        Simple class to overcome a large "switch" needed for all different time patches.
        All functions here are very much the same, besides the unique requirements
        of every type of file (i.e. timepatch).
        :param case: The timepatch value
        :param data_range:
        :param df: Incoming data
        :return: Outgoing DF with proper columns
        """
        assert isinstance(case, str)
        assert isinstance(df, pd.DataFrame)

        df_after = self.__choice_table[case](data_range, df)
        return df_after

