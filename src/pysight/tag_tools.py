"""
__author__ = Hagai Hargil
"""
import numpy as np
import pandas as pd
import warnings
from numba import jit


def verify_periodicity(tag_data: pd.Series=None, tag_freq: float = 0, binwidth: float = 0) -> pd.Series:
    """
    Verify the integrity of the TAG lens pulses periodicity.
    :return: Sorted Series with the new pulses in the right place
    """
    # if not tag_freq

    period = np.ceil(1 / tag_freq / binwidth).astype(np.int32)  # Period of TAG lens in bins
    jitter = 0.05
    allowed_noise = np.ceil(jitter * period).astype(np.uint64)

    # Iteration #0 of while loop:
    tags_diff = np.abs(tag_data.diff() - period)
    tags_diff.loc[tags_diff < allowed_noise] = 0
    missing_ticks = np.where(tags_diff != 0)[0][1:]  # returns a tuple
    changed_ticks = missing_ticks.shape[0]

    if changed_ticks > 0.2 * tag_data.shape[0]:  # Corrupted data
        warnings.warn('TAG Lens data was corrupted. Stack will be created without it.')
        return -1

    # Iterate until TAG pulses are completely periodic
    while changed_ticks != 0:
        new_ticks = np.take(tag_data.values, missing_ticks - 1) + period
        tag_data = tag_data.append(pd.Series(new_ticks, dtype=np.uint64), ignore_index=True)
        tag_data.sort_values(inplace=True)

        tags_diff = np.abs(tag_data.diff() - period)
        tags_diff.loc[tags_diff < allowed_noise] = 0
        missing_ticks = np.where(tags_diff != 0)[0][1:] # returns a tuple
        changed_ticks = missing_ticks.shape[0]

    # Add pulses that didn't exist to make sure all photons indeed have a phase
    last_pulse = tag_data.iat[-1]
    for idx in range(1, 10):
        tag_data = tag_data.append(pd.Series([last_pulse + (idx * period)], dtype=np.uint64), ignore_index=True)

    return tag_data


@jit(nopython=True, cache=True)
def numba_digitize(values: np.array, bins: np.array) -> np.array:
    """
    Numba'd version of np.digitize.
    """
    bins = np.digitize(values, bins)
    relevant_bins = bins > 0
    return bins, relevant_bins


@jit(nopython=True, cache=True)
def numba_find_phase(photons: np.array, bins: np.array, raw_tag: np.array) -> np.array:
    """
    Find the phase [0, 2pi) of the photon for each event in `photons`.
    :return: Numpy array with the size of photons containing the phases.
    """

    phase_vec = np.zeros_like(photons, dtype=np.float32)
    tag_diff = np.diff(raw_tag)

    for idx, bin in enumerate(bins) :  # values of indices that changed
        phase_vec[idx] = (photons[idx] - raw_tag[bin - 1])/tag_diff[bin - 1]
        phase_vec[idx] *= 2 * np.pi

    return phase_vec


def define_phase(df_photons: pd.DataFrame=None, tag_data: pd.Series=None) -> pd.DataFrame:
    """
    Assign each data point its corresponding phase between 0 and 2pi, where max value has a phase of 2pi.
    """

    bin_idx, relevant_bins = numba_digitize(df_photons['abs_time'].values, tag_data.values)
    photons = df_photons['abs_time'].values.astype(float)
    photons[np.logical_not(relevant_bins)] = np.nan
    relevant_photons = np.compress(relevant_bins, photons)

    phase_vec = numba_find_phase(photons=relevant_photons, bins=np.compress(relevant_bins, bin_idx),
                                 raw_tag=tag_data.values)
    first_relevant_photon_idx = photons.shape[0] - relevant_photons.shape[0]
    photons[first_relevant_photon_idx:] = phase_vec

    df_photons['Phase'] = photons
    df_photons.dropna(how='any', inplace=True)

    assert df_photons['Phase'].any() >= 0
    assert df_photons['Phase'].any() < 2 * np.pi

    return df_photons


def interpolate_tag(df_photons: pd.DataFrame=None, tag_data: pd.Series=None, gui=None) -> pd.DataFrame:
    """
    If a TAG channel was defined determine each photon's phase and insert it into the main DataFrame.
    :param df_photons: DataFrame to be changed - TAG phase will be added as index
    :param tag_data: Time of TAG pulses
    :return: Original dataframe with an added index corresponding to each photon's phase
    """


    # TODO: Input verification
    # TODO: Get rid of assumption that the TAG pulse comes at a phase of 0
    # TODO: Still only a single channel of data

    tag_data = verify_periodicity(tag_data=tag_data, tag_freq=float(gui.tag_freq.get()),
                                  binwidth=float(gui.binwidth.get()))
    if isinstance(tag_data, pd.Series):
        df_photons = define_phase(df_photons=df_photons, tag_data=tag_data)

    return df_photons









