"""
Created on Thu Oct 13 09:37:02 2016

__author__: Hagai Hargil
"""


def main_data_readout(gui):
    """
    Main function that reads the lst file and processes its data.
    """
    from pysight.fileIO_tools import FileIO
    from pysight.lst_tools import Analysis
    from pysight.movie_tools import Movie
    from pysight import timepatch_switch
    from pysight.censor_tools import CensorCorrection
    from pysight.output_tools import generate_output_list

    # Read the file
    cur_file = FileIO(filename=gui.filename.get(), debug=gui.debug.get(), input_start=gui.input_start.get(),
                      input_stop1=gui.input_stop1.get(), input_stop2=gui.input_stop2.get())
    cur_file.run()

    # Create input structures
    dict_of_slices_hex = timepatch_switch.ChoiceManagerHex().process(cur_file.timepatch)
    # dict_of_slices_bin = timepatch_switch.ChoiceManagerBinary().process(cur_file.timepatch)  # Not supported

    # Process events into dataframe
    analyzed_struct = Analysis(timepatch=cur_file.timepatch, data_range=cur_file.data_range,
                               dict_of_inputs=cur_file.dict_of_input_channels, data=cur_file.data,
                               is_binary=cur_file.is_binary, num_of_frames=gui.num_of_frames.get(),
                               x_pixels=int(gui.x_pixels.get()), y_pixels=int(gui.y_pixels.get()),
                               laser_freq=float(gui.reprate.get()), binwidth=float(gui.binwidth.get()),
                               dict_of_slices_hex=dict_of_slices_hex, dict_of_slices_bin=None,
                               bidir=gui.bidir.get(), tag_freq=float(gui.tag_freq.get()),
                               tag_pulses=int(gui.tag_pulses.get()), phase=gui.phase.get(),
                               keep_unidir=gui.keep_unidir.get())
    analyzed_struct.run()

    # Create a movie object
    final_movie = Movie(data=analyzed_struct.df_allocated, x_pixels=int(gui.x_pixels.get()),
                        y_pixels=int(gui.y_pixels.get()), z_pixels=int(gui.z_pixels.get()),
                        reprate=float(gui.reprate.get()), name=gui.filename.get(),
                        binwidth=float(gui.binwidth.get()), bidir=gui.bidir.get(),
                        fill_frac=gui.fill_frac.get())

    # Censor correction part
    censored = CensorCorrection(movie=final_movie, reprate=gui.reprate.get(),
                                binwidth=gui.binwidth.get(), offset=5,
                                all_laser_pulses=analyzed_struct.dict_of_data['Laser'])

    # Find out what the user wanted and output it
    print('======================================================= \nOutputs:\n--------')
    output_list = generate_output_list(final_movie, gui)

    return analyzed_struct.df_allocated, final_movie, output_list, censored


def run():
    """
    Run the entire script.
    """
    from pysight.tkinter_gui_multiscaler import GUIApp
    from pysight.tkinter_gui_multiscaler import verify_gui_input

    gui = GUIApp()
    gui.root.mainloop()
    verify_gui_input(gui)
    return main_data_readout(gui)


if __name__ == '__main__':
    df, movie, outputs, censored  = run()
