"""
__author__ = Hagai Hargil
"""
import pathlib
from pysight.main_multiscaler_readout import main_data_readout
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight.tkinter_gui_multiscaler import verify_gui_input

config_filename = r''

path = pathlib.Path(config_filename).parent
all_lst_files = path.glob('*.lst')

gui = GUIApp()
gui.root.mainloop()
verify_gui_input(gui)
df_list = []
movie_list = []
outputs_list = []
censored_list = []

for idx, lst_file in enumerate(all_lst_files):
    df, movie, outputs, censored = main_data_readout(gui)
    df_list.append(df)
    movie_list.append(movie)
    outputs_list.append(outputs)
    censored_list.append(censored)
    label = 6
    data, labels = censored.learn_histograms(label)
    filename = r'/data/Lior/Multiscaler data/06 June 2017/TrainedWeights/17p_label_{}.npy'.format(label)
    import numpy as np

    with open(filename, 'wb') as f:
        np.save(f, data)


