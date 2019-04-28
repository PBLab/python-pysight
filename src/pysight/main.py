"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = Hagai Har-Gil
"""

from typing import Tuple, Dict, Any, Optional
import pickle
import logging
import pathlib
import sys

logging.basicConfig(
    stream=sys.stdout,
    # filename=str(pathlib.Path('.') / "general.log"),
    # filemode="w",
    format="%(levelname)s :: %(filename)s :: %(asctime)s :: %(message)s",
    level=logging.INFO,
)

import matplotlib
import pandas as pd
import colorama
import numpy as np

matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt
import toml

colorama.init()
from pysight.ascii_list_file_parser.file_io import ReadMeta
from pysight.ascii_list_file_parser.tabulation import Tabulate
from pysight.nd_hist_generator.allocation import Allocate
from pysight.nd_hist_generator.movie import Movie
from pysight.ascii_list_file_parser import timepatch_switch
from pysight.nd_hist_generator.outputs import OutputParser, PySightOutput
from pysight.nd_hist_generator.gating import GatedDetection
from pysight.nd_hist_generator.photon_df import PhotonDF
from pysight.nd_hist_generator.tag_bits import ParseTAGBits
from pysight.ascii_list_file_parser.distribute_data import DistributeData
from pysight.nd_hist_generator.line_signal_validators.validation_tools import (
    SignalValidator,
)
from pysight.nd_hist_generator.line_signal_validators.add_bidir_lines import (
    add_bidir_lines,
)
from pysight.gui.gui_main import GuiAppLst
from pysight.gui.gui_helpers import verify_input
from pysight.gui.config_parser import Config
from pysight.nd_hist_generator.volume_gen import VolumeGenerator
from pysight.binary_list_file_parser.binary_parser import BinaryDataParser
from pysight.read_lst import ReadData
from pysight.nd_hist_generator.deinterleave import Deinterleave


def main_data_readout(config: Dict[str, Any]) -> Optional[PySightOutput]:
    """
    Main function that reads the lst file and processes its data.
    Should not be run independently - only from other "run_X" functions.

    :param dict config: Loaded configuration file as a dictionary.

    :return PySightOutput: An object containing the relevant data,\
    if "memory" option was checked in the GUI.
    """
    # Read the .lst file
    if config["outputs"]["data_filename"].endswith(".lst"):
        cur_file = ReadMeta(
            filename=config["outputs"]["data_filename"],
            input_start=config["inputs"]["start"],
            input_stop1=config["inputs"]["stop1"],
            input_stop2=config["inputs"]["stop2"],
            input_stop3=config["inputs"]["stop3"],
            input_stop4=config["inputs"]["stop4"],
            input_stop5=config["inputs"]["stop5"],
            binwidth=config["advanced"]["binwidth"],
            use_sweeps=config["advanced"]["sweeps_as_lines"],
            mirror_phase=config["advanced"]["phase"],
        )
        cur_file.run()
        raw_data_obj = ReadData(
            filename=config["outputs"]["data_filename"],
            start_of_data_pos=cur_file.start_of_data_pos,
            timepatch=cur_file.timepatch,
            is_binary=cur_file.is_binary,
            debug=config["advanced"]["debug"],
        )
        raw_data = raw_data_obj.read_lst()
        if cur_file.is_binary:
            binary_parser = BinaryDataParser(
                data=raw_data,
                data_range=cur_file.data_range,
                timepatch=cur_file.timepatch,
                use_tag_bits=config["tagbits"]["tag_bits"],
                dict_of_inputs=cur_file.dict_of_input_channels,
            )
            binary_parser.run()
        else:
            # Create input structures and create a DataFrame
            dict_of_slices_hex = timepatch_switch.ChoiceManagerHex().process(
                cur_file.timepatch
            )
            tabulated_data = Tabulate(
                data_range=cur_file.data_range,
                data=raw_data,
                dict_of_inputs=cur_file.dict_of_input_channels,
                use_tag_bits=config["tagbits"]["tag_bits"],
                dict_of_slices_hex=dict_of_slices_hex,
                time_after_sweep=cur_file.time_after,
                acq_delay=cur_file.acq_delay,
                num_of_channels=cur_file.num_of_channels,
            )
            tabulated_data.run()

            del raw_data
            del raw_data_obj
            separated_data = DistributeData(
                df=tabulated_data.df_after_timepatch,
                dict_of_inputs=tabulated_data.dict_of_inputs,
                use_tag_bits=config["tagbits"]["tag_bits"],
            )
            separated_data.run()

    ####### START OF "PUBLIC API" ##########
    try:  # list file branch
        if cur_file.is_binary:
            relevant_columns = binary_parser.data_to_grab
            dict_of_data = binary_parser.dict_of_data
        else:
            relevant_columns = separated_data.data_to_grab
            dict_of_data = separated_data.dict_of_data
        lst_metadata = cur_file.lst_metadata
        fill_frac = (
            config["advanced"]["fill_frac"]
            if cur_file.fill_fraction == -1
            else cur_file.fill_fraction
        )
    except NameError:  # dealing with a pickle file
        logging.info(f"Reading file {config['outputs']['data_filename']}...")
        with open(config["outputs"]["data_filename"], "rb") as f:
            dict_of_data = pickle.load(f)
        lst_metadata = dict()
        relevant_columns = ["abs_time"]
        fill_frac = config["advanced"]["fill_frac"]
        binary_parser = None  # to be deleted later

    validated_data = SignalValidator(
        dict_of_data=dict_of_data,
        num_of_frames=config["image"]["num_of_frames"],
        binwidth=float(config["advanced"]["binwidth"]),
        use_sweeps=config["advanced"]["sweeps_as_lines"],
        delay_between_frames=float(config["advanced"]["frame_delay"]),
        data_to_grab=relevant_columns,
        line_freq=config["advanced"]["line_freq"],
        num_of_lines=config["image"]["x_pixels"],
        bidir=config["advanced"]["bidir"],
        bidir_phase=config["advanced"]["phase"],
        image_soft=config["image"]["imaging_software"],
    )

    validated_data.run()
    try:
        del separated_data
    except UnboundLocalError:
        del binary_parser

    photon_df = PhotonDF(
        dict_of_data=validated_data.dict_of_data,
        interleaved=config["advanced"]["interleaved"],
    )
    photons = photon_df.run()
    tag_bit_parser = ParseTAGBits(
        dict_of_data=validated_data.dict_of_data,
        photons=photons,
        use_tag_bits=config["tagbits"]["tag_bits"],
        bits_dict=config["tagbits"],
    )

    if not config["advanced"]["bidir"]:
        validated_data.dict_of_data = add_bidir_lines(validated_data.dict_of_data)
    analyzed_struct = Allocate(
        bidir=config["advanced"]["bidir"],
        tag_offset=config["advanced"]["tag_offset"],
        laser_freq=float(config["advanced"]["reprate"]),
        binwidth=float(config["advanced"]["binwidth"]),
        tag_pulses=int(config["advanced"]["tag_pulses"]),
        phase=config["advanced"]["phase"],
        keep_unidir=config["advanced"]["keep_unidir"],
        flim=config["advanced"]["flim"],
        censor=config["advanced"]["censor"],
        dict_of_data=validated_data.dict_of_data,
        df_photons=tag_bit_parser.gen_df(),
        tag_freq=float(config["advanced"]["tag_freq"]),
        tag_to_phase=True,
        deinterleave=config["advanced"]["interleaved"],
    )
    analyzed_struct.run()
    data_for_movie = analyzed_struct.df_photons

    del photons
    del photon_df
    if config["advanced"]["interleaved"]:
        logging.warning(
            """Deinterleaving a data channel is currently highly experimental and
            is supported only on data in the PMT1 channel. Inexperienced users
            are highly advised not to use it."""
        )
        deinter = Deinterleave(
            photons=analyzed_struct.df_photons,
            reprate=config["advanced"]["reprate"],
            binwidth=config["advanced"]["binwidth"],
        )
        data_for_movie = deinter.run()
    # Determine type and shape of wanted outputs, and open the file pointers there
    outputs = OutputParser(
        num_of_frames=len(validated_data.dict_of_data["Frames"]),
        output_dict=config["outputs"],
        filename=config["outputs"]["data_filename"],
        x_pixels=config["image"]["x_pixels"],
        y_pixels=config["image"]["y_pixels"],
        z_pixels=config["image"]["z_pixels"] if analyzed_struct.tag_interp_ok else 1,
        channels=data_for_movie.index.levels[0],
        flim=config["advanced"]["flim"] or config["advanced"]["interleaved"],
        binwidth=config["advanced"]["binwidth"],
        reprate=config["advanced"]["reprate"],
        lst_metadata=lst_metadata,
        debug=config["advanced"]["debug"],
    )
    outputs.run()

    line_delta = validated_data.line_delta
    del validated_data
    if config["advanced"]["gating"]:
        logging.warning(
            "Gating is currently not implemented. Please contact package authors."
        )
        # gated = GatedDetection(
        #     raw=analyzed_struct.df_photons, reprate=config['advanced']['reprate'], binwidth=config['advanced']['binwidth']
        # )
        # gated.run()

    # Create a movie object
    volume_chunks = VolumeGenerator(
        frames=analyzed_struct.dict_of_data["Frames"], data_shape=outputs.data_shape
    )
    frame_slices = volume_chunks.create_frame_slices()
    final_movie = Movie(
        data=data_for_movie,
        frames=analyzed_struct.dict_of_data["Frames"],
        frame_slices=frame_slices,
        num_of_frame_chunks=volume_chunks.num_of_chunks,
        reprate=float(config["advanced"]["reprate"]),
        name=config["outputs"]["data_filename"],
        data_shape=outputs.data_shape,
        binwidth=float(config["advanced"]["binwidth"]),
        bidir=config["advanced"]["bidir"],
        fill_frac=fill_frac,
        outputs=outputs.outputs,
        censor=config["advanced"]["censor"],
        mirror_phase=config["advanced"]["phase"],
        lines=analyzed_struct.dict_of_data["Lines"],
        channels=data_for_movie.index.levels[0],
        flim=config["advanced"]["flim"] or config["advanced"]["interleaved"],
        lst_metadata=lst_metadata,
        exp_params=analyzed_struct.exp_params,
        line_delta=int(line_delta),
        use_sweeps=config["advanced"]["sweeps_as_lines"],
        tag_as_phase=True,
        tag_freq=float(config["advanced"]["tag_freq"]),
        image_soft=config["image"]["imaging_software"],
        frames_per_chunk=volume_chunks.frames_per_chunk,
    )

    final_movie.run()
    if "memory" in outputs.outputs:
        pysight_output = PySightOutput(
            photons=data_for_movie,
            summed_mem=final_movie.summed_mem,
            stack=final_movie.stack,
            channels=data_for_movie.index.levels[0],
            data_shape=outputs.data_shape,
            flim=config["advanced"]["flim"],
            config=config,
        )
        return pysight_output


def mp_main_data_readout(config: Dict[str, Any]):
    """
    Wrapper for main_data_readout that
    wraps it with a try block. To be used with the
    multiprocessing run option.
    """
    try:
        out = main_data_readout(config)
    except:
        pass
    else:
        return out


def run(cfg_file: str = None) -> Optional[PySightOutput]:
    """ Run PySight.

    :param str cfg_file: Optionally supply an existing configuration filename. Otherwise a GUI will open.

    :return PySightOutput: Object containing raw and processed data
    """
    if cfg_file:
        with open(cfg_file, "r") as f:
            config: Dict[str, Any] = toml.load(f)
    else:
        gui = GuiAppLst()
        gui.root.mainloop()
        config = Config.from_gui(gui).config_data
    verify_input(config)
    return main_data_readout(config)


def run_batch_lst(
    foldername: str,
    glob_str: str = "*.lst",
    recursive: bool = False,
    cfg_file: str = "",
) -> pd.DataFrame:
    """
    Run PySight on all list files in the folder

    :param str foldername: - Main folder to run the analysis on.
    :param str glob_str: String for the `glob` function to filter list files
    :param bool recursive: Whether the search should be recursive.
    :param str cfg_file: Name of config file to use
    :return pd.DataFrame: Record of analyzed data
    """
    path = pathlib.Path(foldername)
    num_of_files = 0
    if not path.exists():
        raise UserWarning(f"Folder {foldername} doesn't exist.")
    if recursive:
        all_lst_files = path.rglob(glob_str)
        logging.info(f"Running PySight on the following files:")
        for file in list(all_lst_files):
            logging.info(str(file))
            num_of_files += 1
        all_lst_files = path.rglob(glob_str)
    else:
        all_lst_files = path.glob(glob_str)
        logging.info(f"Running PySight on the following files:")
        for file in list(all_lst_files):
            logging.info(str(file))
            num_of_files += 1
        all_lst_files = path.glob(glob_str)

    data_columns = ["fname", "done", "error"]
    data_record = pd.DataFrame(
        np.zeros((num_of_files, 3)), columns=data_columns
    )  # store result of PySight
    try:
        with open(cfg_file, "r") as f:
            config = toml.load(f)
            config["outputs"]["data_filename"] = ".lst"
    except (TypeError, FileNotFoundError):
        gui = GuiAppLst()
        gui.root.mainloop()
        gui.filename.set(".lst")  # no need to choose a list file
        config = Config.from_gui(gui).config_data
    verify_input(config)

    try:
        for idx, lst_file in enumerate(all_lst_files):
            config["outputs"]["data_filename"] = str(lst_file)
            data_record.loc[idx, "fname"] = str(lst_file)
            try:
                main_data_readout(config)
            except BaseException as e:
                logging.warning(
                    f"File {str(lst_file)} returned an error. Moving onwards."
                )
                data_record.loc[idx, "done"] = False
                data_record.loc[idx, "error"] = repr(e)
            else:
                data_record.loc[idx, "done"] = True
                data_record.loc[idx, "error"] = None
    except TypeError as e:
        logging.error(repr(e))

    logging.info(f"Summary of batch processing:\n{data_record}")
    return data_record


def mp_batch(
    foldername, glob_str="*.lst", recursive=False, n_proc=None, cfg_file: str = ""
):
    """
    Run several instances of PySight using the multiprocessing module.

    :param str foldername: Folder to scan
    :param str glob_str: Glob string to filter files
    :param bool recursive: Whether to scan subdirectories as well
    :param int n_proc: Number of processes to use (None means all)
    :param str cfg_file: Configuration file name
    """
    import pathlib
    import multiprocessing as mp

    path = pathlib.Path(foldername)
    if not path.exists():
        raise UserWarning(f"Folder {foldername} doesn't exist.")
    if recursive:
        all_lst_files = path.rglob(glob_str)
    else:
        all_lst_files = path.glob(glob_str)
    logging.info(f"Running PySight on the following files:")
    for file in list(all_lst_files):
        logging.info(str(file))

    all_lst_files = path.rglob(glob_str) if recursive else path.glob(glob_str)
    try:
        with open(cfg_file, "r") as f:
            config = toml.load(f)
            config["outputs"]["data_filename"] = ".lst"
    except (TypeError, FileNotFoundError):
        gui = GuiAppLst()
        gui.root.mainloop()
        gui.filename.set(".lst")  # no need to choose a list file
        config = Config.from_gui(gui).config_data
    verify_input(config)
    all_cfgs = []
    for file in all_lst_files:
        config["outputs"]["data_filename"] = str(file)
        all_cfgs.append(config)
    with mp.Pool(n_proc) as pool:
        pool.map(mp_main_data_readout, all_cfgs)


if __name__ == "__main__":
    out = run()
