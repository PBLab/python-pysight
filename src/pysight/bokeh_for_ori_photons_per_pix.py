"""
__author__ = Hagai Hargil
"""
from bokeh.plotting import output_file, show, figure
from bokeh.models import Div
from bokeh.layouts import gridplot, column
import numpy as np

def f(movie):
    hist_p, _ = np.histogram(np.ravel(movie.stack[1][9]), bins=range(15))
    hist_p_summed, _ = np.histogram(np.ravel(movie.summed_mem[1]), bins=range(15))

    one_frame = figure(title="Single Frame")
    one_frame.image(image=[movie.stack[1][9]], x=None, y=None, dw=[1], dh=[1])

    output_file(r"C:\Users\Hagai\to_ori_29-8-17.html", title="Histograms of photons")

    summed_frames = figure(title="Sum of Ten Frames")
    summed_frames.image(image=[movie.summed_mem[1]], x=None, y=None, dw=[1], dh=[1])

    hist_one = figure(y_axis_type="log")
    hist_one.vbar(x=range(14), width=0.5, bottom=1, top=hist_p)
    hist_one.yaxis.axis_label = "Number of pixels"
    hist_one.xaxis.axis_label = "Number of photons"

    hist_summed = figure(y_axis_type="log")
    hist_summed.vbar(x=range(14), width=0.5, bottom=1, top=hist_p_summed)
    hist_summed.xaxis.axis_label = "Number of photons"

    s = gridplot([[one_frame, summed_frames], [hist_one, hist_summed]])
    show(column(Div(text="<h2>Photon distributions, about 0.01 photons per laser pulse</h2>", width=1000),
                s))
