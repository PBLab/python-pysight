from typing import Dict, Any
import pathlib

import toml
import attr
from attr.validators import instance_of


@attr.s
class Config:
    """
    Load, parse and save a TOML config file.
    """

    config_data = attr.ib(validator=instance_of(dict))

    @classmethod
    def from_gui(cls, gui):
        config_data = {
            "cfg_title": gui.save_as.get(),
            "inputs": {
                "stop1": gui.input_stop1.get(),
                "stop2": gui.input_stop2.get(),
                "stop3": gui.input_stop3.get(),
                "stop4": gui.input_stop4.get(),
                "stop5": gui.input_stop5.get(),
                "start": gui.input_start.get(),
            },
            "image": {
                "num_of_frames": gui.num_of_frames.get(),
                "x_pixels": gui.x_pixels.get(),
                "y_pixels": gui.y_pixels.get(),
                "z_pixels": gui.z_pixels.get(),
                "imaging_software": gui.imaging_software.get(),
            },
            "outputs": {
                "data_filename": gui.filename.get(),
                "summed": gui.summed.get(),
                "memory": gui.memory.get(),
                "stack": gui.stack.get(),
                "flim": gui.flim.get(),
            },
            "advanced": {
                "debug": gui.debug.get(),
                "phase": gui.phase.get(),
                "reprate": gui.reprate.get(),
                "gating": gui.gating.get(),
                "binwidth": gui.binwidth.get(),
                "tag_freq": gui.tag_freq.get(),
                "tag_pulses": gui.tag_pulses.get(),
                "tag_offset": gui.tag_offset.get(),
                "fill_frac": gui.fill_frac.get(),
                "bidir": gui.bidir.get(),
                "keep_unidir": gui.keep_unidir.get(),
                "flim": gui.flim.get(),
                "censor": gui.censor.get(),
                "line_freq": gui.line_freq.get(),
                "sweeps_as_lines": gui.sweeps_as_lines.get(),
                "frame_delay": gui.frame_delay.get(),
                "interleaved": gui.interleaved.get(),
            },
            "tagbits": {
                "tag_bits": gui.tag_bits.get(),
                "group1": {
                    "label1": gui.bits_grp_1_label.get(),
                    "start1": gui.bits_grp_1_start.get(),
                    "end1": gui.bits_grp_1_end.get(),
                },
                "group2": {
                    "label2": gui.bits_grp_2_label.get(),
                    "start2": gui.bits_grp_2_start.get(),
                    "end2": gui.bits_grp_2_end.get(),
                },
                "group3": {
                    "label3": gui.bits_grp_3_label.get(),
                    "start3": gui.bits_grp_3_start.get(),
                    "end3": gui.bits_grp_3_end.get(),
                },
            },
        }
        return cls(config_data)

    @classmethod
    def from_disk(cls, fname: pathlib.Path):
        with open(fname, "r") as f:
            config_data = toml.load(f)
        return cls(config_data)

    def to_disk(self):
        """ Write the config_data object to disk """
        full_cfg_fname = (
            pathlib.Path(__file__).resolve().parents[1]
            / "configs"
            / (self.config_data["cfg_title"] + ".toml")
        )
        with open(full_cfg_fname, "w") as f:
            toml.dump(self.config_data, f)
