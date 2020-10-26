from typing import List

import numpy as np
import pandas as pd


from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def _exp_decay(x, a, b, c):
    """ Exponential function for FLIM and censor correction """
    return a * np.exp(-b * x) + c


def add_downsample_frame_idx_to_df(
    data: pd.DataFrame, chan: int, frames: pd.Series, num_of_frames=256
):
    """
    Downsampling in time:
    Receives {data} and {frames} (series which maps line to frame)
    and adds a frame_idx column to {data}. a sequential {num_of_frames} will receive that same frame_idx value
    so that it can later be used to aggregate over {num_of_frames} at a time
    :return: original {data} DF with frame_idx column
    """
    data["frame_idx"] = 0
    for frame_idx in range(len(frames) // num_of_frames):
        start_frame = frames.array[frame_idx * num_of_frames]
        end_frame = frames.array[num_of_frames + frame_idx * num_of_frames - 1]
        data.loc[pd.IndexSlice[chan, start_frame:end_frame], "frame_idx"] = frame_idx
    # add any leftover frames
    if len(frames) // num_of_frames > 0 and len(frames) % num_of_frames > 0:
        data.loc[pd.IndexSlice[chan, end_frame:], "frame_idx"] = (
            len(frames) // num_of_frames
        )

    return data


def add_bins_to_df(df: pd.DataFrame, edges: List[np.ndarray], col_names: List[str]):
    """
    Essentially a multi-dimensions histogram within a pandas dataframe.
    Sets each dimension a bin_of_dim{{num_of_dimension}} column where the value is the bin of the photon.
    Binning is based on {{edges}}.
    Used to group photons by pseudo-coordinates in order to estimate lifetime of a coordinate within a frame.
    :return:
    """

    # add each photon bin in the time
    for ndim, col in zip(range(len(df)), col_names):
        idx = np.searchsorted(edges[ndim], df[col], side="right")
        df["bin_of_dim" + str(ndim)] = idx - 1

    # delete photons outside of the last bin (required due to np.searchsorted side='right')
    for dim, edge in enumerate(edges):
        df = df.loc[df["bin_of_dim" + str(dim)] < len(edge) - 1, :]
        df = df.loc[df["bin_of_dim" + str(dim)] > -1, :]

    return df


def flip_photons(
    data: pd.DataFrame,
    edges_of_lines_dim: np.array,
    lines: np.array,
    num_of_pixels: int,
):
    """
    Receives {data} and flips every odd line by finding the mean time diff between the first 256 lines of {data},
    and subtracting that mean from the odd lines.
    """
    # add each photon bin in the lines dimension
    idx = np.searchsorted(edges_of_lines_dim, data["abs_time"], side="right")
    data["bin_of_dim0"] = idx - 1
    mean_line_diff = lines.diff().iloc[:num_of_pixels].mean().astype(int)
    odd_lines = data.loc[data["bin_of_dim0"] % 2 == 1, "time_rel_line"]
    data.loc[data["bin_of_dim0"] % 2 == 1, "time_rel_line"] = mean_line_diff - odd_lines

    # clean up
    data.drop(columns=["bin_of_dim0"])
    return data


def calc_lifetime(data, bins_bet_pulses=125) -> float:
    """
    calculate lifetime by creating an histogramming the photons by time of arrival,
    we would then expect to see an exponential decay to which we can fit an exponential curve to estimate the lifetime.
    a Savitzky Golay filter is applied to the histogram to smooth out noise.
    :param data: array of photon times of arrival
    :param bins_bet_pulses: number of bins to split the photons into
    :return: estimate lifetime (int)
    """
    hist, edges = np.histogram(data, bins_bet_pulses)
    if (len(data) < 20) or (hist.max() < 5):
        # photon count is too small
        return np.nan

    hist = np.roll(hist, -hist.argmax())
    sg_hist = savgol_filter(hist, 9, 2)

    # find bin with least amount of photons
    lowest_bin_idx = hist.argmin()
    if lowest_bin_idx < 5:
        return np.nan
    sg_hist = sg_hist[:lowest_bin_idx]

    # estimate tau
    try:
        popt, _ = curve_fit(
            _exp_decay,
            np.arange(len(sg_hist)),
            sg_hist,
            p0=[sg_hist.max(), 1 / np.average(sg_hist), sg_hist.min()],
            maxfev=1_000,
            bounds=(0, [len(data), 70, np.inf]),
        )
        tau = 1 / popt[1]
    except (RuntimeError, ValueError):
        tau = np.nan

    if not (70.0 > tau > 0.0001):
        return np.nan

    return tau
