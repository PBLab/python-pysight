import numpy as np
import pandas as pd

from pysight.nd_hist_generator.deinterleave import Deinterleave


class TestDeinterleave:
    photons = np.array([1, 5, 6, 15, 4, 5, 11, 15, 3, 11, 9], dtype=np.uint8)
    abs_time = np.random.randint(100, 1000, photons.shape, dtype=np.uint64)
    df = (
        pd.DataFrame(
            {
                "time_rel_pulse": photons,
                "abs_time": abs_time,
                "Channel": np.ones_like(photons),
                "Lines": np.ones_like(photons),
                "Frames": np.zeros_like(photons),
            }
        )
        .assign(Channel=lambda x: x.Channel.astype("category"))
        .assign(Channel=lambda x: x.Channel.cat.add_categories(7))
        .set_index(["Channel", "Lines", "Frames"], drop=True)
    )

    deinter = Deinterleave(df, reprate=0.25, binwidth=0.25)
    deinter_photons = deinter.run()

    def test_no_photon_left_behind(self):
        assert len(self.deinter_photons) == len(self.photons)

    def test_correct_allocation_early(self):
        idx_slice = pd.IndexSlice
        early_photons = self.deinter_photons.loc[idx_slice[1, :, :], :].copy()
        assert np.array_equal(
            early_photons["time_rel_pulse"].values.ravel(),
            np.array([1, 5, 6, 4, 5, 3], dtype=np.uint8),
        )

    def test_correct_allocation_late(self):
        idx_slice = pd.IndexSlice
        late_photons = self.deinter_photons.loc[idx_slice[7, :, :], :].copy()
        assert np.array_equal(
            late_photons["time_rel_pulse"].values.ravel(),
            np.array([15, 11, 15, 11, 9], dtype=np.uint8),
        )

    def test_abs_time_unchanged(self):
        assert np.array_equal(
            self.deinter_photons["abs_time"].values.ravel().sort(), self.abs_time.sort()
        )
