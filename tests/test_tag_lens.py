import numpy as np
import pandas as pd
from pysight.nd_hist_generator.tag_lens import (
    TagPeriodVerifier,
    TagPhaseAllocator,
    TagPipeline,
    numba_digitize,
)


class TestTagPipeline:
    """
    Tests for TAG analysis functions
    """

    tag_data = pd.Series(np.arange(0, 200 * 6530, 6530))
    photons = pd.DataFrame([10, 100, 1000], columns=["abs_time"])
    def_pipe = TagPipeline(photons=photons, tag_pulses=tag_data)

    def test_preservation(self):
        photons = pd.DataFrame([0, 10, 6531], columns=["abs_time"])
        pipe = TagPipeline(photons=photons, tag_pulses=self.tag_data)
        returned = pd.Series([0, 6530])
        assert np.array_equal(
            returned, pipe._TagPipeline__preserve_relevant_tag_pulses()
        )

    def test_preservation_without_zero(self):
        photons = pd.DataFrame([10, 6531], columns=["abs_time"])
        pipe = TagPipeline(photons=photons, tag_pulses=self.tag_data)
        returned = pd.Series([6530])
        assert np.array_equal(
            returned, pipe._TagPipeline__preserve_relevant_tag_pulses()
        )


class TestTagPeriodVerifier:
    """ Test the Verifier class """

    tag_data = pd.Series(np.arange(0, 200 * 6530, 6530))
    freq = 189e3
    binwidth = 800e-12
    def_verifier = TagPeriodVerifier(
        tag=tag_data, freq=freq, binwidth=binwidth, last_photon=np.uint64(200 * 6530)
    )

    def test_bins_bet_pulses(self):
        assert 6614 == self.def_verifier.period

    def test_allowed_noise(self):
        assert 331 == self.def_verifier.allowed_noise

    def test_start_end_no_issues(self):
        tag_data = pd.Series(np.arange(0, 100, 10))
        freq = 0.1
        binwidth = 1.0
        verifier = TagPeriodVerifier(
            tag=tag_data, freq=freq, binwidth=binwidth, last_photon=np.uint64(100)
        )
        ret_start, ret_end = verifier._TagPeriodVerifier__obtain_start_end_idx()
        my_start = np.array([], dtype=np.int64)
        my_end = np.array([], dtype=np.int64)
        assert np.array_equal(my_start, ret_start)
        assert np.array_equal(my_end, ret_end)

    def test_start_end_no_zero(self):
        tag_data = pd.Series(np.arange(0, 300, 10))
        tag_data.drop([0, 5, 6], inplace=True)
        tag_data = tag_data.append(pd.Series([3, 9, 25]))
        tag_data = tag_data.sort_values().reset_index(drop=True)
        freq = 0.1
        binwidth = 1.0
        verifier = TagPeriodVerifier(
            tag=tag_data, freq=freq, binwidth=binwidth, last_photon=np.uint64(300)
        )
        ret_start, ret_end = verifier._TagPeriodVerifier__obtain_start_end_idx()
        my_start = [0, 3, 6]
        my_end = [2, 5, 7]
        assert list(ret_start) == my_start
        assert list(ret_end) == my_end

    def test_start_end_adding_zero(self):
        tag_data = pd.Series(np.arange(5, 300, 10))
        tag_data.drop([1, 5, 7, 8], inplace=True)
        tag_data = tag_data.append(pd.Series([9, 27, 29, 31]))
        tag_data = tag_data.sort_values().reset_index(drop=True)
        freq = 0.1
        binwidth = 1.0
        verifier = TagPeriodVerifier(
            tag=tag_data, freq=freq, binwidth=binwidth, last_photon=np.uint64(300)
        )
        ret_start, ret_end = verifier._TagPeriodVerifier__obtain_start_end_idx()
        my_start = [0, 7]
        my_end = [6, 9]
        assert list(ret_start) == my_start
        assert list(ret_end) == my_end

    def test_fix_tag_pulses_adding_zero(self):
        tag_data = pd.Series(np.arange(0, 100, 10))
        tag_data.drop([0, 5, 6], inplace=True)
        tag_data = tag_data.append(pd.Series([3, 9, 25]))
        tag_data = tag_data.sort_values().reset_index(drop=True)
        freq = 0.1
        binwidth = 1.0
        verifier = TagPeriodVerifier(
            tag=tag_data, freq=freq, binwidth=binwidth, last_photon=np.uint64(100)
        )
        my_start = [0, 3, 6]
        my_end = [2, 5, 7]
        verifier._TagPeriodVerifier__fix_tag_pulses(starts=my_start, ends=my_end)
        assert np.array_equal(verifier.tag.values, np.arange(0, 100, 10))

    def test_fix_tag_pulses_no_zero_end_missing(self):
        tag_data = pd.Series(np.arange(5, 95, 10, dtype=np.uint64))
        tag_data.drop([1, 5, 7, 8], inplace=True)
        tag_data = tag_data.append(pd.Series([9, 27, 29, 31], dtype=np.uint64))
        tag_data = tag_data.sort_values().reset_index(drop=True)
        freq = 0.1
        binwidth = 1.0
        verifier = TagPeriodVerifier(
            tag=tag_data, freq=freq, binwidth=binwidth, last_photon=np.uint64(85)
        )
        my_start = [0, 7]
        my_end = [6, 8]
        verifier._TagPeriodVerifier__fix_tag_pulses(starts=my_start, ends=my_end)
        assert np.array_equal(verifier.tag.values, np.arange(5, 75, 10))


class TestTagPhaseAllocator:
    def test_tag_digitize(self):
        x = np.array([0.2, 6.4, 3.0, 1.6])
        bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
        real_result = np.array([1, 4, 3, 2])
        result, _ = numba_digitize(x, bins)
        assert np.array_equal(real_result, result)

    def test_allocate_phase_1(self):
        photons = pd.DataFrame([0, 2 * np.pi, 4 * np.pi], columns=["abs_time"])
        tag = pd.Series([0, 2 * np.pi, 4 * np.pi, 6 * np.pi])
        phaser = TagPhaseAllocator(photons, tag)
        phaser.allocate_phase()
        result = [1, 1, 1]
        for elem1, elem2 in zip(result, phaser.photons.Phase.tolist()):
            assert np.isclose(elem1, elem2)

    def test_allocate_phase_2(self):
        photons = pd.DataFrame([np.pi, 3 * np.pi, 5 * np.pi], columns=["abs_time"])
        tag = pd.Series([0, 2 * np.pi, 4 * np.pi, 6 * np.pi])
        phaser = TagPhaseAllocator(photons, tag)
        phaser.allocate_phase()
        for elem1, elem2 in zip([-1, -1, -1], phaser.photons.Phase.tolist()):
            assert np.isclose(elem1, elem2)

    def test_allocate_phase_3(self):
        photons = pd.DataFrame([1, 2, 3], columns=["abs_time"])
        tag = pd.Series([0, 2 * np.pi, 4 * np.pi, 6 * np.pi])
        phaser = TagPhaseAllocator(photons, tag)
        phaser.allocate_phase()
        normed_result = photons.abs_time / (2 * np.pi)
        true_result = np.sin(normed_result * 2 * np.pi + np.pi / 2).astype(np.float32)
        for elem1, elem2 in zip(true_result.tolist(), phaser.photons.Phase.tolist()):
            assert np.isclose(elem1, elem2, 6)
