import json
import logging
from typing import Union

from pysight.gui.gui_main import GuiAppLst, ImagingSoftware


def verify_gui_input(gui: GuiAppLst):
    """Validate all GUI inputs"""
    data_sources = set(gui.tuple_of_data_sources)
    channel_inputs = {gui.input_start, gui.input_stop1, gui.input_stop2}
    MINIMAL_TAG_BIT = 1
    MAXIMAL_TAG_BIT = 16

    if gui.input_start != "PMT1":
        if gui.input_stop1 != "PMT1":
            if gui.input_stop2 != "PMT1":
                raise BrokenPipeError("PMT1 value has to be entered in inputs.")

    if gui.num_of_frames == None:
        if "Frames" not in data_sources:
            raise BrokenPipeError(
                "You must either enter a frame channel or number of frames."
            )
    else:
        if float(gui.num_of_frames) != gui.num_of_frames:
            raise ValueError("Please enter an integer number of frames.")

    if gui.num_of_frames < 0:
        raise ValueError("Number of frames has to be a positive number.")

    filename = gui.filename
    if not filename.endswith(".lst") and not filename.endswith(".p"):
        raise BrokenPipeError(
            "Please choose a list (*.lst) or pickle (*.p) file for analysis."
        )

    if channel_inputs > data_sources:
        raise ValueError(
            "Wrong inputs in channels. Please choose a value from the list."
        )

    list_of_keys = [gui.input_start, gui.input_stop1, gui.input_stop2]
    set_of_keys = set(list_of_keys)

    if len(list_of_keys) != len(
        set_of_keys
    ):  # making sure only a single option was chosen in the GUI
        if [x for x in list_of_keys if x != "Empty"] != list(
            set_of_keys.difference({"Empty"})
        ):
            raise KeyError(
                'Input consisted of two or more similar names which are not "Empty".'
            )

    # TAG bits input verification
    if gui.tag_bits:
        values_of_bits_set = set()
        start_bits_set = set()
        end_bits_set = set()
        for key, val in gui.tag_bits_dict.items():
            if val.value not in gui.tag_bits_group_options:
                raise UserWarning(f"Value {val} not in allowed TAG bits inputs.")
            if not isinstance(val.start, int):
                raise UserWarning(
                    f"The start bit of TAG label {val.value} wasn't an integer."
                )
            if not isinstance(val.end, int):
                raise UserWarning(
                    f"The end bit of TAG label {val.value} wasn't an integer."
                )
            if val.end < val.start:
                raise UserWarning(
                    f"Bits in row {key + 1} have a start value which is higher than its end."
                )
            if val.start > MAXIMAL_TAG_BIT or val.end > MAXIMAL_TAG_BIT:
                raise UserWarning(
                    f"In label {key} maximal allowed TAG bit is {MAXIMAL_TAG_BIT}."
                )
            if val.start < MINIMAL_TAG_BIT or val.end < MINIMAL_TAG_BIT:
                raise UserWarning(
                    f"In label {key} minimal allowed TAG bit is {MINIMAL_TAG_BIT}."
                )
            values_of_bits_set.add(val)
            start_bits_set.add(val.start)
            end_bits_set.add(val.end)

        if len(values_of_bits_set) > len(start_bits_set):
            raise UserWarning("Some TAG bit labels weren't given unique start bits.")

        if len(values_of_bits_set) > len(end_bits_set):
            raise UserWarning("Some TAG bit labels weren't given unique end bits.")

    if not isinstance(gui.phase, float) and not isinstance(gui.phase, int):
        raise UserWarning("Mirror phase must be a number.")

    if gui.fill_frac < 0:
        raise UserWarning("Fill fraction must be a positive number.")

    if not isinstance(gui.fill_frac, float) and not isinstance(gui.flyback, int):
        raise UserWarning("Fill fraction must be a number.")

    if gui.x_pixels < 0:
        raise UserWarning("X pixels value must be greater than 0.")

    if gui.y_pixels < 0:
        raise UserWarning("X pixels value must be greater than 0.")

    if gui.z_pixels < 0:
        raise UserWarning("X pixels value must be greater than 0.")
    try:
        int(gui.x_pixels)
        int(gui.x_pixels)
        int(gui.x_pixels)
    except ValueError:
        raise UserWarning("Pixels must be an integer number.")

    if float(gui.x_pixels) != gui.x_pixels:
        raise UserWarning("Enter an integer number for the x-axis pixels.")

    if float(gui.y_pixels) != gui.y_pixels:
        raise UserWarning("Enter an integer number for the y-axis pixels.")

    if float(gui.z_pixels) != gui.z_pixels:
        raise UserWarning("Enter an integer number for the z-axis pixels.")

    if gui.reprate < 0:
        raise UserWarning("Laser repetition rate must be positive.")

    if (gui.reprate % 10 != 0) and (gui.flim or gui.interleaved):
        raise UserWarning("FLIM or Deinterleaving isn't supported for laser "
        "reprates that can't be used with a 10 MHz external clock."
        " Please contact package authors.")

    if gui.binwidth < 0:
        raise UserWarning("Binwidth must be a positive number.")

    if gui.binwidth > 1e-9:
        raise UserWarning("Enter a binwidth with units of [seconds].")

    if type(gui.filename) != str:
        raise UserWarning("Filename must be a string.")

    if "Laser" in channel_inputs and gui.flim == 1:
        raise UserWarning(
            "Can't have both a laser channel active and the FLIM checkboxed ticked."
        )

    if not gui.imaging_software.upper() in [
        name for name, member in ImagingSoftware.__members__.items()
    ]:
        raise UserWarning("Must use existing options in the Imaging Software entry.")

    if not isinstance(gui.frame_delay, float):
        raise UserWarning("Frame delay must be a float.")

    if gui.frame_delay < 0:
        raise UserWarning("Frame delay must be a positive float.")

    if gui.frame_delay > 10:
        raise UserWarning(
            "Frame delay is the number of seconds between subsequent frames."
        )


def convert_json_to_input_dict(cfg_fname):
    """ Convert a config file to a usable dictionary """
    if not isinstance(cfg_fname, str):
        raise TypeError

    try:
        with open(cfg_fname) as f:
            gui = json.load(f)
        gui = {key: val[1] for key, val in gui.items()}
        return gui
    except:
        raise TypeError


class GuiHelper:
    """
    Helper class to create intermediate representation
    of the GUI's content.
    """

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.tag_bits_dict = dict(
            bits_grp_1_label=entries["bits_grp_1_label"],
            bits_grp_2_label=entries["bits_grp_2_label"],
            bits_grp_3_label=entries["bits_grp_3_label"],
        )

    @property
    def outputs(self):
        """ Create a dictionary with the wanted user outputs. """
        output = {}

        if self.summed is True:
            output["summed"] = True
        if self.memory is True:
            output["memory"] = True
        if self.stack is True:
            output["stack"] = True

        if "stack" in output:
            if not "summed" in output and not "memory" in output:
                logging.warning(
                    "Performance Warning: Writing data to file might take a long time when the required"
                    " output is only 'Full Stack'."
                )
        return output


def tkinter_to_object(gui: Union[GuiAppLst, dict]) -> GuiHelper:
    """ Convert a tkinter instance into a pickable dictionary """
    if isinstance(gui, dict):
        gui["tuple_of_data_sources"] = (
            "PMT1",
            "PMT2",
            "Lines",
            "Frames",
            "Laser",
            "TAG Lens",
            "Empty",
        )
        return GuiHelper(**gui)
    else:
        dic = {
            key: val.get() for key, val in gui.__dict__.items() if "Var" in repr(val)
        }
        dic["tuple_of_data_sources"] = gui.tuple_of_data_sources
        return GuiHelper(**dic)
