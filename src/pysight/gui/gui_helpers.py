import json
import logging
from typing import Union, MutableMapping, Any

from pysight.nd_hist_generator.movie import ImagingSoftware


def verify_input(config: MutableMapping[str, Any]):
    """Validate all GUI inputs"""
    data_sources = (
        "PMT1",
        "PMT2",
        "PMT3",
        "PMT4",
        "Lines",
        "Frames",
        "Laser",
        "TAG Lens",
        "Empty",
    )
    channel_inputs = {
        config["inputs"]["start"],
        config["inputs"]["stop1"],
        config["inputs"]["stop2"],
        config["inputs"]["stop3"],
        config["inputs"]["stop4"],
        config["inputs"]["stop5"],
    }
    MINIMAL_TAG_BIT = 1
    MAXIMAL_TAG_BIT = 16

    list_of_keys = [
        config["inputs"]["start"],
        config["inputs"]["stop1"],
        config["inputs"]["stop2"],
        config["inputs"]["stop3"],
        config["inputs"]["stop4"],
        config["inputs"]["stop5"],
    ]
    short_list_of_keys = [x[:-1] for x in list_of_keys]
    if "PMT" not in short_list_of_keys:
        raise BrokenPipeError("At least one PMT channel has to be entered in inputs.")

    if config["image"]["num_of_frames"] == None:
        if "Frames" not in data_sources:
            raise BrokenPipeError(
                "You must either enter a frame channel or number of frames."
            )
    else:
        if float(config["image"]["num_of_frames"]) != config["image"]["num_of_frames"]:
            raise ValueError("Please enter an integer number of frames.")

    if config["image"]["num_of_frames"] < 0:
        raise ValueError("Number of frames has to be a positive number.")

    filename = config["outputs"]["data_filename"]
    if (
        not filename.endswith(".lst")
        and not filename.endswith(".npz")
        and not filename.endswith("*")
    ):
        raise BrokenPipeError(
            "Please choose a list (*.lst) or numpy archive (*.npz) file for analysis."
        )

    if len(channel_inputs) > len(data_sources):
        raise ValueError(
            "Wrong inputs in channels. Please choose a value from the list."
        )

    set_of_keys = set(list_of_keys)

    if len(list_of_keys) != len(
        set_of_keys
    ):  # making sure only a single option was chosen in the GUI
        non_empty_keys = sorted([x for x in list_of_keys if x != "Empty"])
        sorted_set_of_keys = sorted(list(set_of_keys.difference({"Empty"})))
        if non_empty_keys != sorted_set_of_keys:
            raise KeyError(
                'Input consisted of two or more similar names which are not "Empty".'
            )

    # TAG bits input verification
    tag_bits_group_options = (
        "Power",
        "Slow axis",
        "Fast axis",
        "Z axis",
        "None",
    )
    if config["tagbits"]["tag_bits"]:
        values_of_bits_set = set()
        start_bits_set = set()
        end_bits_set = set()
        for key, val in config["tagbits"].items():
            try:
                groupnum = key[-1]
            except ValueError:
                continue
            cur_label = val["label" + groupnum]
            if cur_label not in tag_bits_group_options:
                raise UserWarning(f"Value {val} not in allowed TAG bits inputs.")
            if not isinstance(val["start" + groupnum], int):
                raise UserWarning(
                    f"The start bit of TAG label {val['start' + groupnum]} wasn't an integer."
                )
            if not isinstance(val["end"] + groupnum, int):
                raise UserWarning(
                    f"The end bit of TAG label {val['end' + groupnum]} wasn't an integer."
                )
            if val["end" + groupnum] < val["start" + groupnum]:
                raise UserWarning(
                    f"Bits in row {groupnum} have a start value which is higher than its end."
                )
            if (
                val["start" + groupnum] > MAXIMAL_TAG_BIT
                or val["end" + groupnum] > MAXIMAL_TAG_BIT
            ):
                raise UserWarning(
                    f"In label {key} maximal allowed TAG bit is {MAXIMAL_TAG_BIT}."
                )
            if (
                val["start" + groupnum] < MINIMAL_TAG_BIT
                or val["end" + groupnum] < MINIMAL_TAG_BIT
            ):
                raise UserWarning(
                    f"In label {key} minimal allowed TAG bit is {MINIMAL_TAG_BIT}."
                )
            values_of_bits_set.add(val["label" + groupnum])
            start_bits_set.add(val["start" + groupnum])
            end_bits_set.add(val["end" + groupnum])

        if len(values_of_bits_set) > len(start_bits_set):
            raise UserWarning("Some TAG bit labels weren't given unique start bits.")

        if len(values_of_bits_set) > len(end_bits_set):
            raise UserWarning("Some TAG bit labels weren't given unique end bits.")

    if not isinstance(config["advanced"]["phase"], float) and not isinstance(
        config["advanced"]["phase"], int
    ):
        raise UserWarning("Mirror phase must be a number.")

    if not isinstance(config["advanced"]["flim_downsampling_space"], int):
        raise UserWarning("Downsampling in space must be a number.")

    if not isinstance(config["advanced"]["flim_downsampling_time"], int):
        raise UserWarning("Downsampling in time must be a number.")

    if config["advanced"]["fill_frac"] < 0:
        raise UserWarning("Fill fraction must be a positive number.")

    if not isinstance(config["advanced"]["fill_frac"], float) and not isinstance(
        config["advanced"]["fill_frac"], int
    ):
        raise UserWarning("Fill fraction must be a number.")

    try:
        int(config["image"]["x_pixels"])
        int(config["image"]["x_pixels"])
        int(config["image"]["x_pixels"])
    except ValueError:
        raise UserWarning("Pixels must be an integer number.")

    if config["image"]["x_pixels"] < 0:
        raise UserWarning("X pixels value must be greater than 0.")

    if config["image"]["y_pixels"] < 0:
        raise UserWarning("X pixels value must be greater than 0.")

    if config["image"]["z_pixels"] < 0:
        raise UserWarning("X pixels value must be greater than 0.")

    if float(config["image"]["x_pixels"]) != config["image"]["x_pixels"]:
        raise UserWarning("Enter an integer number for the x-axis pixels.")

    if float(config["image"]["y_pixels"]) != config["image"]["y_pixels"]:
        raise UserWarning("Enter an integer number for the y-axis pixels.")

    if float(config["image"]["z_pixels"]) != config["image"]["z_pixels"]:
        raise UserWarning("Enter an integer number for the z-axis pixels.")

    if config["advanced"]["reprate"] < 0:
        raise UserWarning("Laser repetition rate must be positive.")

    if (config["advanced"]["reprate"] % 10 != 0) and (
        config["advanced"]["flim"] or config["advanced"]["interleaved"]
    ):
        raise UserWarning(
            "FLIM or Deinterleaving isn't supported for laser "
            "reprates that can't be used with a 10 MHz external clock."
            " Please contact package authors."
        )

    if config["advanced"]["binwidth"] < 0:
        raise UserWarning("Binwidth must be a positive number.")

    if config["advanced"]["binwidth"] > 1e-9:
        raise UserWarning("Enter a binwidth with units of [seconds].")

    if type(config["outputs"]["data_filename"]) != str:
        raise UserWarning("Filename must be a string.")

    if "Laser" in channel_inputs and config["advanced"]["flim"] == 1:
        raise UserWarning(
            "Can't have both a laser channel active and the FLIM checkboxed ticked."
        )

    if not config["image"]["imaging_software"].upper() in [
        name for name, member in ImagingSoftware.__members__.items()
    ]:
        raise UserWarning("Must use existing options in the Imaging Software entry.")

    if not isinstance(config["advanced"]["frame_delay"], float):
        raise UserWarning("Frame delay must be a float.")

    if config["advanced"]["frame_delay"] < 0:
        raise UserWarning("Frame delay must be a positive float.")

    if config["advanced"]["frame_delay"] > 10:
        raise UserWarning(
            "Frame delay is the number of seconds between subsequent frames."
        )
