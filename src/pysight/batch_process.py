"""
__author__ = Hagai Hargil
"""
import pathlib
from pysight.main_multiscaler_readout import main_data_readout
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight.tkinter_gui_multiscaler import verify_gui_input


def batch_lst_files(foldername):
    """ Analyze all .lst files in a dir with a single cfg file """

    path = pathlib.Path(foldername)
    all_lst_files = path.glob('*.lst')

    gui = GUIApp()
    gui.root.mainloop()
    verify_gui_input(gui)

    for idx, lst_file in enumerate(all_lst_files):
        if lst_file.stat().st_size > 3e2:
            gui.filename.set(str(lst_file))
            df, movie = main_data_readout(gui)
            print(f"File {str(lst_file)} analyzed successfully, moving onwards...")
        else:
            print(f"File {str(lst_file)} was empty.")
