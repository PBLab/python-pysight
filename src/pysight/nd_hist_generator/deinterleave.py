import attr
from attrs.validator import instance_of
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
    late_photons = attr.ib(init=False)


    @property
    def bins_bet_pulses(self):
        return int(np.ceil(1/self.reprate / self.binwidth))

    def run(self):
        """ Main pipeline for this class """
        # TODO
