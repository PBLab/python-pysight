import pandas as pd
import numpy as np
import attr
from typing import Dict
from attr.validators import instance_of


@attr.s(slots=True)
class DistributeData:
    """
    Separates the channel-specific data to their own channels.
    Inputs:
        :param df: pd.DataFrame with data
        :param dict_of_inputs: Mapping of inputs to data they contain
        :param use_tag_bits: Whether TAG bits are needed
    """
    df = attr.ib(validator=instance_of(pd.DataFrame))
    dict_of_inputs = attr.ib(validator=instance_of(dict))
    use_tag_bits = attr.ib(default=False, validator=instance_of(bool))
    dict_of_data = attr.ib(init=False)
    data_to_grab = attr.ib(init=False)

    def run(self) -> None:
        """ Runs the allocation function, populating self.dict_of_data """
        self.dict_of_data = self.__allocate_data_by_channel()

    def __allocate_data_by_channel(self) -> Dict:
        """
        Go over the channels and find the events from that specific channel, assigning
        them to a dictionary with a suitable name.
        :return: Dict containing the data
        """
        dict_of_data = {}
        self.data_to_grab = ['abs_time', 'sweep']  # relevant columns of the DF for analysis
        if self.use_tag_bits:
            self.data_to_grab.extend(['tag', 'edge'])
        for key in self.dict_of_inputs:
            relevant_values = self.df.loc[self.df['channel'] == self.dict_of_inputs[key], self.data_to_grab]
            if key in ['PMT1', 'PMT2']:
                dict_of_data[key] = relevant_values.reset_index(drop=True)
                dict_of_data[key]['Channel'] = 1 if 'PMT1' == key else 2  # channel is the spectral channel
            else:
                dict_of_data[key] = relevant_values.sort_values(by=['abs_time']).reset_index(drop=True)

        return dict_of_data
