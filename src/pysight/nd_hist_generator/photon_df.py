import attr
from attr.validators import instance_of
import pandas as pd
import sys


@attr.s(slots=True)
class PhotonDF:
    """
    Create initial photon dataframe and set the channel as its index. It will
    contain a PMT\photons column for every recorded data channel.

    :param Dict[str, pd.DataFrame] dict_of_data: A dictionary with keys
    being the input analog channels and values are the DataFrames that contains the raw data.
    :param int num_of_channels: Number of data channels that will be generated.
    """

    dict_of_data = attr.ib(validator=instance_of(dict))
    num_of_channels = attr.ib(init=False)

    def run(self) -> pd.DataFrame:
        """
        Main function for the class.
        Concatenates all valid data channels into a single dataframe.

        :return pd.DataFrame: Photon data with labeled channels.
        """
        self.num_of_channels = 1
        if "Channel" not in self.dict_of_data["PMT1"].columns:
            self.dict_of_data["PMT1"]["Channel"] = 1

        try:
            if "Channel" not in self.dict_of_data["PMT2"].columns:
                self.dict_of_data["PMT2"]["Channel"] = 2
        except KeyError:
            pass

        try:
            df_photons = pd.concat(
                [self.dict_of_data["PMT1"].copy(), self.dict_of_data["PMT2"].copy()],
                axis=0,
            )
            self.num_of_channels += 1
        except KeyError:
            df_photons = self.dict_of_data["PMT1"].copy()
        except:
            print("Unknown error: ", sys.exc_info()[0])
        finally:
            df_photons.loc[:, "Channel"] = df_photons.loc[:, "Channel"].astype(
                "category"
            )
            df_photons.set_index(keys="Channel", inplace=True)

        return df_photons
