"""
__author__ = Hagai Hargil
"""
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight import fileIO_tools
from pysight import lst_tools
from pysight import timepatch_switch

gui = GUIApp()
gui.root.mainloop()
data_range = fileIO_tools.get_range(gui.filename.get())
timepatch = fileIO_tools.get_timepatch(gui.filename.get())
start_of_data_pos = fileIO_tools.get_start_pos(gui.filename.get())
dict_of_input_channels = fileIO_tools.create_inputs_dict(gui=gui)
list_of_recorded_data_channels = fileIO_tools.find_active_channels(gui.filename.get())
fileIO_tools.compare_recorded_and_input_channels(dict_of_input_channels, list_of_recorded_data_channels)

# Read the file into a variable
print('Reading file {}'.format(gui.filename.get()))
dict_of_slices = timepatch_switch.ChoiceManager().process(timepatch)

data = fileIO_tools.read_lst(filename=gui.filename.get(), start_of_data_pos=start_of_data_pos,
                             timepatch=timepatch)

df_after_timepatch = lst_tools.tabulate_input(data=data, dict_of_slices=dict_of_slices, data_range=data_range,
                                              input_channels=dict_of_input_channels)
print('Input tabulated.')
# dict_of_data = lst_tools.determine_data_channels(df=df_after_timepatch,
#                                                  dict_of_inputs=dict_of_input_channels,
#                                                  num_of_frames=int(gui.num_of_frames.get()),
#                                                  x_pixels=int(gui.x_pixels.get()),
#                                                  y_pixels=int(gui.y_pixels.get()),
#                                                  laser_freq=float(gui.reprate.get()),
#                                                  binwidth=float(gui.binwidth.get()))
