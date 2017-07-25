"""
__author__ = Hagai Hargil
"""
import pathlib
from pysight.main_multiscaler_readout import main_data_readout
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight.tkinter_gui_multiscaler import verify_gui_input

config_filename = r'X:\Lior\Multiscaler data\24 July 2017\No HoldAfterSweep'

path = pathlib.Path(config_filename)
all_lst_files = path.glob('*.lst')

gui = GUIApp()
gui.root.mainloop()
verify_gui_input(gui)
df_list = []
movie_list = []
outputs_list = []
censored_list = []

for idx, lst_file in enumerate(all_lst_files):
    if lst_file.stat().st_size > 3e3:
        gui.filename.set(str(lst_file))
        df, movie = main_data_readout(gui)
    else:
        print(f"File {str(lst_file)} was empty.")


