"""
__author__ = Hagai Hargil
"""
import attr
from attr.validators import instance_of
import pandas as pd
import sys


@attr.s(slots=True)
class PhotonDF(object):
    """
    Create initial photon dataframe and set the channel as its index
    """
    dict_of_data    = attr.ib(validator=instance_of(dict))
    num_of_channels = attr.ib(default=1, validator=instance_of(int))

    def gen_df(self):
        """
        If a single PMT channel exists, create a df_photons object.
        Else, concatenate the two data channels into a single dataframe.
        :return pd.DataFrame: Photon data
        """

        assert 'Channel' in self.dict_of_data['PMT1'].columns

        try:
            df_photons = pd.concat([self.dict_of_data['PMT1'].copy(),
                                    self.dict_of_data['PMT2'].copy()], axis=0)
            self.num_of_channels = 2
        except KeyError:
            df_photons = self.dict_of_data['PMT1'].copy()
        except:
            print("Unknown error: ", sys.exc_info()[0])
        finally:
            df_photons.loc[:, 'Channel'] = df_photons.loc[:, 'Channel'].astype('category')
            df_photons.set_index(keys='Channel', inplace=True)

        return df_photons


