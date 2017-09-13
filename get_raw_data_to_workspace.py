"""
__author__ = Hagai Hargil
"""
from pysight.tkinter_gui_multiscaler import GUIApp, verify_gui_input
from pysight.fileIO_tools import FileIO
from pysight.tabulation_tools import Tabulate
from pysight.allocation_tools import Allocate
from pysight.tag_bits_tools import ParseTAGBits
from pysight.photon_df_tools import PhotonDF
from pysight import timepatch_switch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gui = GUIApp()
    gui.root.mainloop()
    verify_gui_input(gui)

    # Read the file
    cur_file = FileIO(filename=gui.filename.get(), debug=gui.debug.get(), input_start=gui.input_start.get(),
                      input_stop1=gui.input_stop1.get(), input_stop2=gui.input_stop2.get(), binwidth=gui.binwidth.get(),
                      use_sweeps=gui.sweeps_as_lines.get())
    cur_file.run()

    # Create input structures
    dict_of_slices_hex = timepatch_switch.ChoiceManagerHex().process(cur_file.timepatch)
    # dict_of_slices_bin = timepatch_switch.ChoiceManagerBinary().process(cur_file.timepatch)  # Not supported

    # Process events into dataframe
    tabulated_data = Tabulate(timepatch=cur_file.timepatch, data_range=cur_file.data_range,
                              dict_of_inputs=cur_file.dict_of_input_channels, data=cur_file.data,
                              is_binary=cur_file.is_binary, num_of_frames=gui.num_of_frames.get(),
                              laser_freq=float(gui.reprate.get()), binwidth=float(gui.binwidth.get()),
                              dict_of_slices_hex=dict_of_slices_hex, dict_of_slices_bin=None,
                              use_tag_bits=gui.tag_bits.get(), use_sweeps=gui.sweeps_as_lines.get(),
                              time_after_sweep=cur_file.time_after, acq_delay=cur_file.acq_delay,
                              line_freq=gui.line_freq.get(), x_pixels=gui.x_pixels.get(),
                              y_pixels=gui.y_pixels.get())
    tabulated_data.run()

    photon_df = PhotonDF(dict_of_data=tabulated_data.dict_of_data)
    tag_bit_parser = ParseTAGBits(dict_of_data=tabulated_data.dict_of_data, photons=photon_df.gen_df(),
                                  use_tag_bits=gui.tag_bits.get(), bits_dict=gui.tag_bits_dict)

    analyzed_struct = Allocate(dict_of_inputs=cur_file.dict_of_input_channels,
                               laser_freq=float(gui.reprate.get()), binwidth=float(gui.binwidth.get()),
                               tag_pulses=int(gui.tag_pulses.get()), phase=gui.phase.get(),
                               keep_unidir=gui.keep_unidir.get(), flim=gui.flim.get(),
                               censor=gui.censor.get(), dict_of_data=tabulated_data.dict_of_data,
                               df_photons=tag_bit_parser.gen_df())
    analyzed_struct.run()
    print(f"Number of raw photons: {tabulated_data.dict_of_data['PMT1'].shape[0]}.")
    print(f"Number of processed photons: {analyzed_struct.df_photons.shape[0]}.")
    raw_diffs = tabulated_data.dict_of_data['PMT1'].abs_time.diff()
    plt.figure()
    raw_diffs.hist(bins=16)
    plt.figure()
    analyzed_struct.df_photons.time_rel_pulse.hist(bins=16)
