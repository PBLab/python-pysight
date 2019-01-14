import numpy as np
import pandas as pd

from pysight.nd_hist_generator.deinterleave import Deinterleave

class TestDeinterleave:
    df = pd.DataFrame({'time_rel_pulse': [1, 5, 6, 15, 4, 5, 11]})
