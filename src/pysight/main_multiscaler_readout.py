"""
Created on Thu Oct 13 09:37:02 2016

__author__: Hagai Hargil
"""
from pysight.tkinter_gui_multiscaler import GUIApp
from pysight.tkinter_gui_multiscaler import verify_gui_input
from pysight.output_tools import generate_output_list


def main_data_readout(gui):
    """
    Main function that reads the lst file and processes its data.
    """
    from pysight import fileIO_tools
    from pysight import lst_tools
    from pysight import class_defs
    from pysight import timepatch_switch

    # Open file and find the needed parameters
    is_binary = fileIO_tools.determine_binary(gui.filename.get())
    timepatch = fileIO_tools.get_timepatch(gui.filename.get(), is_binary)
    if timepatch == '3' and not is_binary:
        raise NotImplementedError('Timepatch value "3" is currently not supported for hex files. Please message the package owner.')
    data_range = fileIO_tools.get_range(gui.filename.get(), is_binary)

    start_of_data_pos = fileIO_tools.get_start_pos(gui.filename.get(), is_binary)
    dict_of_input_channels = fileIO_tools.create_inputs_dict(gui=gui)
    list_of_recorded_data_channels = fileIO_tools.find_active_channels(gui.filename.get(), is_binary)
    fileIO_tools.compare_recorded_and_input_channels(dict_of_input_channels,
                                                     list_of_recorded_data_channels)  # TODO: Warn about Empty channels

    # Read the file into a variable
    if gui.debug.get() == 0:
        num_of_items = -1
        read_string = 'Reading file "{}"...'.format(gui.filename.get())
    else:
        num_of_items = 1e6
        read_string = '[DEBUG] Reading file "{}"...'.format(gui.filename.get())

    print(read_string)
    data = fileIO_tools.read_lst(filename=gui.filename.get(), start_of_data_pos=start_of_data_pos,
                                 timepatch=timepatch, num_of_items=num_of_items, is_binary=is_binary)

    print('File read. Sorting the file according to timepatch...')
    if is_binary:
        dict_of_slices = timepatch_switch.ChoiceManagerBinary().process(timepatch)
        df_after_timepatch = lst_tools.tabulate_input_binary(data=data, dict_of_slices=dict_of_slices,
                                                             data_range=data_range,
                                                             input_channels=dict_of_input_channels)
    else:
        dict_of_slices = timepatch_switch.ChoiceManagerHex().process(timepatch)
        df_after_timepatch = lst_tools.tabulate_input_hex(data=data, dict_of_slices=dict_of_slices,
                                                          data_range=data_range,
                                                          input_channels=dict_of_input_channels)

    print('Sorted dataframe created. Starting setting the proper data channel distribution...')

    # Assign the proper channels to their data and function
    dict_of_data, line_delta = lst_tools.determine_data_channels(df=df_after_timepatch,
                                                                 dict_of_inputs=dict_of_input_channels,
                                                                 num_of_frames=int(gui.num_of_frames.get()),
                                                                 x_pixels=int(gui.x_pixels.get()),
                                                                 y_pixels=int(gui.y_pixels.get()),
                                                                 laser_freq=float(gui.reprate.get()),
                                                                 binwidth=float(gui.binwidth.get()),
                                                                 flyback=gui.flyback.get())
    print('Channels of events found. Allocating photons to their frames and lines...')

    df_allocated = lst_tools.allocate_photons(dict_of_data=dict_of_data, gui=gui, line_delta=line_delta)
    print('Relative times calculated. Creating Movie object...')

    # Create a movie object
    final_movie = class_defs.Movie(data=df_allocated, x_pixels=int(gui.x_pixels.get()),
                                   y_pixels=int(gui.y_pixels.get()), z_pixels=int(gui.z_pixels.get()),
                                   reprate=float(gui.reprate.get()), name=gui.filename.get(),
                                   binwidth=float(gui.binwidth.get()), bidir=gui.bidir.get())

    # Find out what the user wanted and output it
    print('======================================================= \nOutputs:\n--------')
    output_list = generate_output_list(final_movie, gui)
    return df_allocated, final_movie, output_list


def run():
    """
    Run the entire script.
    """
    gui = GUIApp()
    gui.root.mainloop()
    verify_gui_input(gui)
    df_after, movie_after, list_of_outputs = main_data_readout(gui)
    return df_after, movie_after, list_of_outputs

if __name__ == '__main__':
    df, movie, outputs = run()



