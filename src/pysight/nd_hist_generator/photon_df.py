import attr
from attr.validators import instance_of
import pandas as pd
import sys
import logging


@attr.s(slots=True)
class PhotonDF:
    """
    Create initial photon dataframe and set the channel as its index. It will
    contain a PMT/photons column for every recorded data channel.

    :param Dict[str, pd.DataFrame] dict_of_data: A dictionary with keys
    being the input analog channels and values are the DataFrames that contains the raw data.
    :param bool interleaved: Whether the PMT1 channel is interleaved.
    """

    dict_of_data = attr.ib(validator=instance_of(dict))
    interleaved = attr.ib(validator=instance_of(bool))

    def run(self) -> pd.DataFrame:
        """
        Main function for the class.
        Concatenates all valid data channels into a single dataframe.

        :return pd.DataFrame: Photon data with labeled channels.
        """
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
        except KeyError:
            df_photons = self.dict_of_data["PMT1"].copy()
        except:
            logging.error("Unknown error: ", sys.exc_info()[0])
        finally:
            df_photons.loc[:, "Channel"] = df_photons.loc[:, "Channel"].astype(
                "category"
            )
            if self.interleaved:
                df_photons["Channel"] = df_photons["Channel"].cat.add_categories(7)
            df_photons.set_index(keys="Channel", inplace=True)

        return df_photons
