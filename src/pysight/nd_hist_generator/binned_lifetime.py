import builtins
import numpy as np
from operator import index
from collections import namedtuple
from numba import jit, prange, objmode
from math import gamma
from scipy.optimize import curve_fit

__all__ = [
    'binned_lifetime_dd']
BinnedStatisticddResult = namedtuple('BinnedStatisticddResult',
                                     ('statistic', 'bin_edges',
                                      'binnumber'))


@jit(nopython=True)
def _numba_convolve_mode_valid(arr, kernel):
    # workaround for running np.convolve with mode='valid' in numba
    n = kernel.size
    return np.convolve(arr, kernel)[n-1:1-n]


@jit(nopython=True)
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
   The Savitzky-Golay filter removes high frequency noise from data.
   It has the advantage of preserving the original shape and
   features of the signal better than other types of filtering
   approaches, such as moving averages techniques.
   Parameters
   ----------
   y : array_like, shape (N,)
       the values of the time history of the signal.
   window_size : int
       the length of the window. Must be an odd integer number.
   order : int
       the order of the polynomial used in the filtering.
       Must be less then `window_size` - 1.
   deriv: int
       the order of the derivative to compute (default = 0 means only smoothing)
   Returns
   -------
   ys : ndarray, shape (N)
       the smoothed signal (or it's n-th derivative).
   Notes
   -----
   The Savitzky-Golay is a type of low-pass filter, particularly
   suited for smoothing noisy data. The main idea behind this
   approach is to make for each point a least-square fit with a
   polynomial of high order over a odd-sized window centered at
   the point.
   Examples
   --------
   t = np.linspace(-4, 4, 500)
   y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
   ysg = savitzky_golay(y, window_size=31, order=4)
   import matplotlib.pyplot as plt
   plt.plot(t, y, label='Noisy signal')
   plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
   plt.plot(t, ysg, 'r', label='Filtered signal')
   plt.legend()
   plt.show()
   References
   ----------
   .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
      Data by Simplified Least Squares Procedures. Analytical
      Chemistry, 1964, 36 (8), pp 1627-1639.
   .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
      W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
      Cambridge University Press ISBN-13: 9780521880688
   """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.array([[float(k**i) for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).astype(np.float32)[deriv] * rate**deriv * gamma(deriv+1)
    # pad the signal at the extremes with
    # values taken from the signal itselfp
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    #return np.convolve( m[::-1], y, mode='valid')[y.size-1:1-y.size]
    return _numba_convolve_mode_valid(y, m[::-1])


def _exp_decay(x, a, b, c):
    """ Exponential function for FLIM and censor correction """
    return a * np.exp(-b * x) + c


@jit(nopython=True, parallel=True)
def _extract_lifetimes_loop(binnumbers, Vdim_range, result, values):
    """
    iterate over bins numbers and dimensions to extract the FLIM lifetime from grouped pixels
    """
    unique_binnumbers = np.unique(binnumbers)
    for idx in prange(unique_binnumbers.min(), unique_binnumbers.max()):
        i = binnumbers[idx]
        for vv in Vdim_range:
            data = values[vv][binnumbers == i]
            lifetime = _calc_lifetime(data)
            result[vv, i] = lifetime if lifetime < 1000 else 1000  # normalize long lifetimes to 1000
    return result


@jit(nopython=True)
def _calc_lifetime(data, bins_bet_pulses=125, bins_lowest_idx_upper_bound=65) -> float:
    """Calculate the lifetime of the given data by fitting it to a decaying exponent
    with a lifetime around 3 ns.
    """
    hist, edges = np.histogram(data, bins_bet_pulses)
    if data.sum() < 50 or hist.max() < 5:
        # photon count is too small
        return 0.0
    hist = np.roll(hist, -hist.argmax())

    sg_hist = savitzky_golay(hist, 9, 2)

    # find bin with least amount of photons under {bins_lowest_idx_upper_bound}
    lowest_bin_idx = hist[:bins_lowest_idx_upper_bound].argmin()
    if lowest_bin_idx < 2:
        return 0.0
    sg_hist = sg_hist[:lowest_bin_idx]

    with objmode(tau='float32'):
        # curve_fit won't work in numba nopython mode
        x = np.linspace(0, len(sg_hist), len(sg_hist))
        popt, _ = curve_fit(_exp_decay, x, sg_hist, p0=[max(sg_hist), 1/np.average(sg_hist), min(sg_hist)],
                            maxfev=10000)
        tau = popt[1]

    return 1/tau


def binned_lifetime_dd(sample, values, bins=10, range=None, expand_binnumbers=False, binned_statistic_result=None):
    """
    Modified scipy.binned_statistic_dd function that executes FLIM lifetime extraction logic with numba.
    """

    try:
        bins = index(bins)
    except TypeError:
        # bins is not an integer
        pass
    # If bins was an integer-like object, now it is an actual Python int.

    # NOTE: for _bin_edges(), see e.g. gh-11365
    if isinstance(bins, int) and not np.isfinite(sample).all():
        raise ValueError('%r contains non-finite values.' % (sample,))

    # `Ndim` is the number of dimensions (e.g. `2` for `binned_statistic_2d`)
    # `Dlen` is the length of elements along each dimension.
    # This code is based on np.histogramdd
    try:
        # `sample` is an ND-array.
        Dlen, Ndim = sample.shape
    except (AttributeError, ValueError):
        # `sample` is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        Dlen, Ndim = sample.shape

    # Store initial shape of `values` to preserve it in the output
    values = np.asarray(values)
    input_shape = list(values.shape)
    # Make sure that `values` is 2D to iterate over rows
    values = np.atleast_2d(values)
    Vdim, Vlen = values.shape

    # Make sure `values` match `sample`
    if(Vlen != Dlen):
        raise AttributeError('The number of `values` elements must match the '
                             'length of each `sample` dimension.')

    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal '
                                 'to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]

    if binned_statistic_result is None:
        nbin, edges, dedges = _bin_edges(sample, bins, range)
        binnumbers = _bin_numbers(sample, nbin, edges, dedges)
    else:
        edges = binned_statistic_result.bin_edges
        nbin = np.array([len(edges[i]) + 1 for i in builtins.range(Ndim)])
        # +1 for outlier bins
        dedges = [np.diff(edges[i]) for i in builtins.range(Ndim)]
        binnumbers = binned_statistic_result.binnumber

    result = _extract_lifetimes_loop(binnumbers, np.array(builtins.range(Vdim)),
                                     np.zeros([Vdim, nbin.prod()], dtype=np.float32), values)
    # Shape into a proper matrix
    result = result.reshape(np.append(Vdim, nbin))

    # Remove outliers (indices 0 and -1 for each bin-dimension).
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]

    # Unravel binnumbers into an ndarray, each row the bins for each dimension
    if(expand_binnumbers and Ndim > 1):
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))

    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')

    # Reshape to have output (`result`) match input (`values`) shape
    result = result.reshape(input_shape[:-1] + list(nbin-2))

    return BinnedStatisticddResult(result, edges, binnumbers)


def _bin_edges(sample, bins=None, range=None):
    """ Create edge arrays
    """
    Dlen, Ndim = sample.shape

    nbin = np.empty(Ndim, int)    # Number of bins in each dimension
    edges = Ndim * [None]         # Bin edges for each dim (will be 2D array)
    dedges = Ndim * [None]        # Spacing between edges (will be 2D array)

    # Select range for each dimension
    # Used only if number of bins is given.
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        smin = np.zeros(Ndim)
        smax = np.zeros(Ndim)
        for i in builtins.range(Ndim):
            smin[i], smax[i] = range[i]

    # Make sure the bins have a finite width.
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - .5
            smax[i] = smax[i] + .5

    # Create edge arrays
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2  # +2 for outlier bins
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1)
        else:
            edges[i] = np.asarray(bins[i], float)
            nbin[i] = len(edges[i]) + 1  # +1 for outlier bins
        dedges[i] = np.diff(edges[i])

    nbin = np.asarray(nbin)

    return nbin, edges, dedges


def _bin_numbers(sample, nbin, edges, dedges):
    """Compute the bin number each sample falls into, in each dimension
    """
    Dlen, Ndim = sample.shape

    sampBin = [
        np.digitize(sample[:, i], edges[i])
        for i in range(Ndim)
    ]

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    for i in range(Ndim):
        # Find the rounding precision
        dedges_min = dedges[i].min()
        if dedges_min == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        decimal = int(-np.log10(dedges_min)) + 6
        # Find which points are on the rightmost edge.
        on_edge = np.where(np.around(sample[:, i], decimal) ==
                           np.around(edges[i][-1], decimal))[0]
        # Shift these points one bin to the left.
        sampBin[i][on_edge] -= 1

    # Compute the sample indices in the flattened statistic matrix.
    binnumbers = np.ravel_multi_index(sampBin, nbin)

    return binnumbers
