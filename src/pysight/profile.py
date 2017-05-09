"""
__author__ = Hagai Hargil
"""
import pstats, cProfile
from pysight.main_multiscaler_readout import main_data_readout
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight.tkinter_gui_multiscaler import verify_gui_input

import pyximport
pyximport.install()

gui = GUIApp()
gui.root.mainloop()
verify_gui_input(gui)
cProfile.runctx("main_data_readout(gui)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
