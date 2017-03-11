"""
__author__ = Hagai Hargil
"""

# Very fast IO that returns a tuple:
import numpy as np
import pandas as pd


start_of_data_pos = 1553
filename=r'C:\Users\Hagai\Documents\GitHub\Multiscaler_Image_Generator\Update\live mouse  100 um deep with 62p TAG010.lst'
with open(filename, "rb") as f:
    f.seek(start_of_data_pos)
    arr = np.fromfile(f, dtype='14S')  # 14 letters

with open(filename, "rb") as f:
    f.seek(start_of_data_pos)
    arr = np.fromfile(f, dtype='14S').tostring().decode()  # Fast
    df = pd.DataFrame(arr)


