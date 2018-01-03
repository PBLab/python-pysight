"""
__author__ = Hagai Hargil
"""
import pandas as pd
import numpy as np
import attr
from typing import Dict
from attr.validators import instance_of


@attr.s(slots=True)
class DistributeData:
    """
    Separates the channel-specific data to their own channels
    # ABOUT TO BE REFACTORED
    """
    df = attr.ib(validator=instance_of(pd.DataFrame))
    dict_of_inputs = attr.ib(validator=instance_of(dict))
    use_tag_bits = attr.ib(default=False, validator=instance_of(bool))
    dict_of_data = attr.ib(init=False)
    data_to_grab = attr.ib(init=False)

    def run(self) -> None:
        """
        Main pipeline
        :return:
        """
        self.dict_of_data = self.__allocate_data_by_channel()

    def __allocate_data_by_channel(self) -> Dict:
        """
        Go over the channels and find the events from that specific channel, assigning
        them to a dictionary with a suitable name.
        :return: Dict containing the data
        """
        dict_of_data = {}
        self.data_to_grab = ['abs_time', 'sweep']
        if self.use_tag_bits:
            self.data_to_grab.extend(['tag', 'edge'])
        for key in self.dict_of_inputs:
            relevant_values = self.df.loc[self.df['channel'] == self.dict_of_inputs[key], self.data_to_grab]
            # NUMBA SORT NOT WORKING:
            # sorted_vals = numba_sorted(relevant_values.values)
            # dict_of_data[key] = pd.DataFrame(sorted_vals, columns=['abs_time'])
            if key in ['PMT1', 'PMT2']:
                dict_of_data[key] = relevant_values.reset_index(drop=True)
                dict_of_data[key]['Channel'] = 1 if 'PMT1' == key else 2  # Channel is the spectral channel
            else:
                dict_of_data[key] = relevant_values.sort_values(by=['abs_time']).reset_index(drop=True)

        return dict_of_data
