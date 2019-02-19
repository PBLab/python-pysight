import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd


@attr.s(slots=True)
class Deinterleave:
    """
    **HIGHLY EXPERIMENTAL**

    This class can differentiate the data recorded in PMT1 into two
    distinct channels by observing the time that has passed between
    the laser pulse and the photon detection time. Early photons
    should be allocated to the first channel, while the latter ones
    will be moved into a new channel, "PMT7".

    This class should be used when two (or more, in the future) beams
    are co-aligned and excite the sample near-simultaneously.
    """

    photons = attr.ib(validator=instance_of(pd.DataFrame))
    reprate = attr.ib(default=80.3e6, validator=instance_of(float))
    binwidth = attr.ib(default=800e-12, validator=instance_of(float))
    num_of_beams = attr.ib(default=2, validator=instance_of(int))

    @property
    def bins_bet_pulses(self):
        return int(np.ceil(1 / self.reprate / self.binwidth))

    def run(self) -> pd.DataFrame:
        """
        Main pipeline for this class. Returns a DataFrame in which
        Channel 1 was deinterleaved into two channels, early and late
        with the late data going into channel 7.
        """
        late_photons_mask = self.photons.xs(1, level=0).mask(
            self.photons["time_rel_pulse"].xs(1, level=0) > (self.bins_bet_pulses // 2)
        )
        early_photons = late_photons_mask.dropna().assign(
            time_rel_pulse=lambda x: x.time_rel_pulse.astype(np.uint8)
        )
        late_photons = self.photons.xs(1, level=0).loc[
            late_photons_mask["time_rel_pulse"].isna(), :
        ]

        # TODO: Change following paragraph when pandas 0.24 hits with Int64 extension type
        for column in early_photons.columns:
            early_photons[column] = early_photons[column].astype(
                late_photons[column].dtype
            )
        remaining_data_channels = self.photons.index.levels[0].categories[1:-1]
        other_channels_photon_df = []
        for chan in remaining_data_channels:
            other_channels_photon_df.append(self.photons.xs(chan, level=0))

        new_photons = (
            pd.concat(
                [early_photons] + other_channels_photon_df + [late_photons],
                keys=self.photons.index.levels[0].categories.values,
                names=["Channel"],
            )
            .reset_index(level=0)
            .assign(Channel=lambda x: x["Channel"].astype("category"))
            .set_index("Channel", append=True)
            .reorder_levels((2, 0, 1))
        )
        return new_photons
