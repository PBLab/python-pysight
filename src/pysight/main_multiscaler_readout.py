# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:37:02 2016

@author: Hagai
"""
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight.tkinter_gui_multiscaler import verify_gui_input
import numpy as np


def main_data_readout(gui):
    """
    Main function that reads the lst file and processes its data.
    """
    from pysight import lst_tools
    from pysight import class_defs

    # Open file and find the needed parameters
    data_range = lst_tools.get_range(gui.filename.get())
    timepatch = lst_tools.get_timepatch(gui.filename.get())
    start_of_data_pos = lst_tools.get_start_pos(gui.filename.get())
    dict_of_input_channels = lst_tools.create_inputs_dict(gui=gui)

    # Read the file into a variable
    print('Reading file...')
    df = lst_tools.read_lst_file(filename=gui.filename.get(), start_of_data_pos=start_of_data_pos)
    print('File read. Sorting the file according to timepatch...')

    # Create a dataframe with all needed columns
    df_after_timepatch = lst_tools.timepatch_sort(df=df, timepatch=timepatch, data_range=data_range,
                                                  input_channels=dict_of_input_channels)
    print('Sorted dataframe created. Starting setting the proper data channel distribution...')

    # Assign the proper channels to their data and function
    dict_of_data = lst_tools.determine_data_channels(df=df_after_timepatch,
                                                     dict_of_inputs=dict_of_input_channels,
                                                     num_of_frames=int(gui.num_of_frames.get()),
                                                     num_of_rows=int(gui.y_pixels.get()))
    print('Channels of events found. Allocating photons to their frames and lines...')

    df_allocated = lst_tools.allocate_photons(dict_of_data=dict_of_data)
    print('Relative times calculated. Creating Movie object...')

    # Create a movie object
    movie = class_defs.Movie(data=df_allocated, num_of_cols=int(gui.x_pixels.get()),
                             num_of_rows=int(gui.y_pixels.get()), reprate=float(gui.reprate.get()),
                             name=gui.filename.get(), binwidth=float(gui.binwidth.get()))
    final_stack = movie.play()
    print('Tiff stack created with name {}.'.format(gui.filename.get()))

    return df_allocated, movie, final_stack

def run():
    """ Run the entire script.
    :return: df_after - dataframe with data
    :return: movie - the Movie object that contains the frames
    :return: final_stack - data of images
    """
    gui = GUIApp()
    gui.root.mainloop()
    verify_gui_input(gui)
    df_after, movie, final_stack = main_data_readout(gui)
    return df_after, movie, final_stack

if __name__ == '__main__':
    df_after, movie, final_stack = run()



