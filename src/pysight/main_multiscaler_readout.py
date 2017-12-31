"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = Hagai Hargil
Created on Thu Oct 13 09:37:02 2016
"""
import pandas as pd


def main_data_readout(gui):
    """
    Main function that reads the lst file and processes its data.
    """
    from pysight.fileIO_tools import FileIO
    from pysight.tabulation_tools import Tabulate
    from pysight.allocation_tools import Allocate
    from pysight.movie_tools import Movie
    from pysight import timepatch_switch
    from pysight.output_tools import OutputParser
    from pysight.gating_tools import GatedDetection
    from pysight.photon_df_tools import PhotonDF
    from pysight.tag_bits_tools import ParseTAGBits
    from pysight.distribute_data import DistributeData
    from pysight.validation_tools import SignalValidator
    import numpy as np

    # Read the file
    cur_file = FileIO(filename=gui.filename.get(), debug=gui.debug.get(), input_start=gui.input_start.get(),
                      input_stop1=gui.input_stop1.get(), input_stop2=gui.input_stop2.get(), binwidth=gui.binwidth.get(),
                      use_sweeps=gui.sweeps_as_lines.get())
    cur_file.run()

    # Create input structures
    dict_of_slices_hex = timepatch_switch.ChoiceManagerHex().process(cur_file.timepatch)
    # dict_of_slices_bin = timepatch_switch.ChoiceManagerBinary().process(cur_file.timepatch)  # Not supported

    # Process events into dataframe
    tabulated_data = Tabulate(data_range=cur_file.data_range, data=cur_file.data,
                              dict_of_inputs=cur_file.dict_of_input_channels,
                              is_binary=cur_file.is_binary, use_tag_bits=gui.tag_bits.get(),
                              dict_of_slices_hex=dict_of_slices_hex, dict_of_slices_bin=None,
                              time_after_sweep=cur_file.time_after, acq_delay=cur_file.acq_delay,
                              num_of_channels=cur_file.num_of_channels, )
    tabulated_data.run()

    separated_data = DistributeData(df=tabulated_data.df_after_timepatch,
                                    dict_of_inputs=tabulated_data.dict_of_inputs,
                                    use_tag_bits=gui.tag_bits.get(), )
    separated_data.run()

    validated_data = SignalValidator(dict_of_data=separated_data.dict_of_data, num_of_frames=gui.num_of_frames.get(),
                                    binwidth=float(gui.binwidth.get()), use_sweeps=gui.sweeps_as_lines.get(),
                                    delay_between_frames=float(gui.frame_delay.get()),
                                    data_to_grab=separated_data.data_to_grab, line_freq=gui.line_freq.get(),
                                    num_of_lines=gui.x_pixels.get(), bidir=gui.bidir.get(),
                                    bidir_phase=gui.phase.get(), image_soft=gui.imaging_software.get(),
                                    )

    validated_data.run()

    photon_df = PhotonDF(dict_of_data=separated_data.dict_of_data)
    tag_bit_parser = ParseTAGBits(dict_of_data=separated_data.dict_of_data, photons=photon_df.gen_df(),
                                  use_tag_bits=gui.tag_bits.get(), bits_dict=gui.tag_bits_dict)

    analyzed_struct = Allocate(dict_of_inputs=cur_file.dict_of_input_channels, bidir=gui.bidir.get(),
                               laser_freq=float(gui.reprate.get()), binwidth=float(gui.binwidth.get()),
                               tag_pulses=int(gui.tag_pulses.get()), phase=gui.phase.get(),
                               keep_unidir=gui.keep_unidir.get(), flim=gui.flim.get(),
                               censor=gui.censor.get(), dict_of_data=separated_data.dict_of_data,
                               df_photons=tag_bit_parser.gen_df(), num_of_channels=tabulated_data.num_of_channels,
                               tag_freq=float(gui.tag_freq.get()), tag_to_phase=True, tag_offset=gui.tag_offset.get())
    analyzed_struct.run()

    # Determine type and shape of wanted outputs, and open the file pointers there
    outputs = OutputParser(num_of_frames=len(np.unique(analyzed_struct.df_photons.index
                                                       .get_level_values('Frames')).astype(np.uint64)),
                           output_dict=gui.outputs, filename=gui.filename.get(),
                           x_pixels=gui.x_pixels.get(), y_pixels=gui.y_pixels.get(),
                           z_pixels=gui.z_pixels.get() if analyzed_struct.tag_interp_ok else 1,
                           num_of_channels=analyzed_struct.num_of_channels, flim=gui.flim.get(),
                           binwidth=gui.binwidth.get(), reprate=gui.reprate.get(),
                           lst_metadata=cur_file.lst_metadata, debug=gui.debug.get())
    outputs.run()

    if gui.gating.get():
        gated = GatedDetection(raw=analyzed_struct.df_photons, reprate=gui.reprate.get(),
                               binwidth=gui.binwidth.get())
        gated.run()

    data_for_movie = gated.data if gui.gating.get() else analyzed_struct.df_photons

    # Create a movie object
    final_movie = Movie(data=data_for_movie, x_pixels=int(gui.x_pixels.get()),
                        y_pixels=int(gui.y_pixels.get()), z_pixels=outputs.z_pixels,
                        reprate=float(gui.reprate.get()), name=gui.filename.get(),
                        binwidth=float(gui.binwidth.get()), bidir=gui.bidir.get(),
                        fill_frac=gui.fill_frac.get() if cur_file.fill_fraction == -1.0 else cur_file.fill_fraction,
                        outputs=outputs.outputs, censor=gui.censor.get(), mirror_phase=gui.phase.get(),
                        lines=validated_data.dict_of_data['Lines'].abs_time,
                        num_of_channels=analyzed_struct.num_of_channels, flim=gui.flim.get(),
                        lst_metadata=cur_file.lst_metadata, exp_params=analyzed_struct.exp_params,
                        line_delta=int(validated_data.line_delta), use_sweeps=gui.sweeps_as_lines.get(),
                        tag_as_phase=True, tag_freq=float(gui.tag_freq.get()), )

    final_movie.run()

    return analyzed_struct.df_photons, final_movie


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


def run_batch(foldername: str, glob_str: str="*.lst", recursive: bool=False) -> pd.DataFrame:
    """
    Run PySight on all list files in the folder
    :param foldername: str - Main folder to run the analysis on.
    :param glob_str: String for the `glob` function to filter list files
    :param recursive: bool - Whether the search should be recursive.
    :return pd.DataFrame: Record of analyzed data
    """

    import pathlib
    from pysight.tkinter_gui_multiscaler import GUIApp
    from pysight.tkinter_gui_multiscaler import verify_gui_input
    import numpy as np

    path = pathlib.Path(foldername)
    num_of_files = 0
    if not path.exists():
        raise UserWarning(f"Folder {foldername} doesn't exist.")
    if recursive:
        all_lst_files = path.rglob(glob_str)
        print(f"Running PySight on the following files:")
        for file in list(all_lst_files):
            print(str(file))
            num_of_files += 1
        all_lst_files = path.rglob(glob_str)
    else:
        all_lst_files = path.glob(glob_str)
        print(f"Running PySight on the following files:")
        for file in list(all_lst_files):
            print(str(file))
            num_of_files += 1
        all_lst_files = path.glob(glob_str)

    data_columns = ['fname', 'done', 'error']
    data_record = pd.DataFrame(np.zeros((num_of_files, 3)), columns=data_columns)  # store result of PySight
    gui = GUIApp()
    gui.root.mainloop()
    gui.filename.set('.lst')  # no need to choose a list file
    verify_gui_input(gui)

    try:
        for idx, lst_file in enumerate(all_lst_files):
            gui.filename.set(str(lst_file))
            data_record.loc[idx, 'fname'] = str(lst_file)
            try:
                main_data_readout(gui)
            except BaseException as e:
                print(f"File {str(lst_file)} returned an error. Moving onwards.")
                data_record.loc[idx, 'done'] = False
                data_record.loc[idx, 'error'] = repr(e)
            else:
                data_record.loc[idx, 'done'] = True
                data_record.loc[idx, 'error'] = None
    except TypeError as e:
        print(repr(e))

    print(f"Summary of batch processing:\n{data_record}")
    return data_record


if __name__ == '__main__':
    df, movie = run()
    # data = run_batch(foldername="", glob_str="*.lst", recursive=False)
