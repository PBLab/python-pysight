"""
__author__ = Hagai Hargil
"""
from bokeh.plotting import output_file, show, figure
import numpy as np


def f(movie):
    # Compute
    hist, _ = np.histogram(movie.data.time_rel_pulse, bins=16)
    A0 = np.max(hist)
    K0, C0 = 1/5, A0/4

    max_idx = np.argmax(hist)
    t = np.arange(max_idx, 16)
    y = hist[max_idx:]
    A, K = fit_exp_linear(t, y, C0)

    # Bokeh setups
    output_file(r"C:\Users\Hagai\to_ori_29-8-17_flim.html", title="FLIM fits")
    fig = figure(title="Exponential fit")
    fig.line(t, y, legend=f"tau={0.8/K} ns")
    fig.vbar(x=range(16), bottom=0, width=0.5, top=hist)

    show(fig)

def fit_exp_linear(t, y, C):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, -K
