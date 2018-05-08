"""
__author__ = Hagai Hargil
"""
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import font as tkfont
import json
from typing import Dict, Union, Tuple, Iterable
from pathlib import Path, WindowsPath
from os import sep, utime
import time
import attr
from attr.validators import instance_of
from enum import Enum
import pysight


def is_positive(instance, attribute, value):
    if value < 0:
        return ValueError("TAG Bit value has to be greater than 0.")


def end_is_greater(instance, attribute, value):
    if value < instance.start:
        return ValueError("TAG Bit 'end' value has to be equal or greater to 'start'.")


class ImagingSoftware(Enum):
    SCANIMAGE = 'ScanImage'
    MSCAN = 'MScan'


@attr.s(slots=True)
class TagBits(object):
    """
    Storage for TAG bits
    """
    value = attr.ib(default="None", validator=instance_of(str))
    start = attr.ib(default=0, validator=[instance_of(int),
                                          is_positive])
    end   = attr.ib(default=1, validator=[instance_of(int),
                                          is_positive,
                                          end_is_greater])

class GuiAppLst(object):
    """
    Main GUI for the multiscaler code.
    Note - class variables should contain "entry" in their name if they point
    to an entry TTK object. Also, no variable should contain "root" in its name.
    """
    def __init__(self):

        self.root = Tk()
        self._data_sources = ('PMT1', 'PMT2', 'Lines', 'Frames', 'Laser', 'TAG Lens', 'Empty')
        self.root.title(f"PySight \uFF5C  PBLab \uFF5C v{pysight.__version__}")
        self.root.rowconfigure(16, weight=1)
        self.root.columnconfigure(16, weight=1)
        main_frame = ttk.Frame(self.root, width=1000, height=1300)
        main_frame.grid(column=0, row=0)
        main_frame['borderwidth'] = 2
        style = ttk.Style()
        style.theme_use('clam')
        self.normal_font = tkfont.Font(family='Helvetica', size=10)
        self.bold_font = tkfont.Font(family='Helvetica', size=12, weight='bold')
        self.config_row = 11
        self.__create_vars()

        # Run widgets
        self.__browse_file(main_frame)
        self.__advanced_win(main_frame)
        self.__input_channels(main_frame)
        self.__num_of_frames(main_frame)
        self.__outputs(main_frame)
        self.__image_size(main_frame)
        self.__tag_bits(main_frame)
        self.__imaging_software(main_frame)

        # Only saving\loading functions after this point
        self.__save_cfg(main_frame)
        self.__load_cfg(main_frame)
        self.__load_last_used_cfg(main_frame)

        # Define the last quit button and wrap up GUI
        quit_button = ttk.Button(main_frame, text='Start', command=self.root.destroy)
        quit_button.grid(row=13, column=4, sticky='ns')

        self.root.bind("<Return>", self.__dest)
        for child in main_frame.winfo_children():
            child.grid_configure(padx=3, pady=2)

        self.root.wait_window()

    def __dest(self, event):
        self.root.destroy()

    def __create_vars(self):
        self.debug = BooleanVar(value=False)
        self.phase = DoubleVar(value=-2.78)
        self.reprate = DoubleVar(value=80.3e6)  # 80e6 for the Chameleon, 0 to raise ZeroDivisionError
        self.gating = BooleanVar(value=False)  # difference between pulse and arrival to sample
        self.binwidth = DoubleVar(value=800e-12)
        self.tag_freq = DoubleVar(value=0.189e6)
        self.tag_pulses = IntVar(value=1)
        self.tag_offset = IntVar(value=0)
        self.fill_frac = DoubleVar(value=72.0)  # percent
        self.bidir = BooleanVar(value=False)
        self.keep_unidir = BooleanVar(value=False)
        self.flim: BooleanVar = BooleanVar(value=False)
        self.censor: BooleanVar = BooleanVar(value=False)
        self.line_freq = DoubleVar(value=7930.0)  # Hz
        self.sweeps_as_lines = BooleanVar(value=False)
        self.frame_delay = DoubleVar(value=0.001)  # sec

    def __browse_file(self, main_frame):
        file_row=0
        self.filename = StringVar(value="")
        browse_button = ttk.Button(main_frame, text="Browse", command=self.__browsefunc)
        browse_button.grid(column=0, row=file_row, sticky='ns')

        browse_entry = ttk.Entry(main_frame, textvariable=self.filename, width=80)
        browse_entry.grid(column=1, row=file_row, sticky='we', columnspan=4)

    def __imaging_software(self, main_frame):
        imaging_software_label = ttk.Label(main_frame, text='Imaging Software', font=self.bold_font)
        imaging_software_label.grid(row=1, column=4, sticky='ns')
        self.imaging_software = StringVar()
        cb_image = ttk.Combobox(main_frame, textvariable=self.imaging_software, width=10)
        cb_image.grid(row=2, column=4, sticky='ns')
        cb_image.set(ImagingSoftware.SCANIMAGE.value)
        cb_image['values'] = [item.value for item in ImagingSoftware]

    def __input_channels(self, main_frame):
        # Comboboxes
        inputs_row = 1
        input_channels_label = ttk.Label(main_frame, text='Input Channels                         ', font=self.bold_font)
        input_channels_label.grid(column=0, row=inputs_row, columnspan=2)
        self.input_start = StringVar()
        self.input_stop1 = StringVar()
        self.input_stop2 = StringVar()
        self.tuple_of_data_sources = self._data_sources
        mb1 = ttk.Combobox(main_frame, textvariable=self.input_start, width=10)
        mb1.grid(column=1, row=inputs_row+1, sticky='w')
        mb1.set('PMT1')
        mb1['values'] = self.tuple_of_data_sources
        mb2 = ttk.Combobox(main_frame, textvariable=self.input_stop1, width=10)
        mb2.grid(column=1, row=inputs_row+2, sticky='w')
        mb2.set('Empty')
        mb2['values'] = self.tuple_of_data_sources
        mb3 = ttk.Combobox(main_frame, textvariable=self.input_stop2, width=10)
        mb3.grid(column=1, row=inputs_row+3, sticky='w')
        mb3.set('Lines')
        mb3['values'] = self.tuple_of_data_sources

        # Labels
        input_channel_1 = ttk.Label(main_frame, text='START', font=self.normal_font)
        input_channel_1.grid(column=0, row=inputs_row+1, sticky='ns')
        input_channel_2 = ttk.Label(main_frame, text='STOP1', font=self.normal_font)
        input_channel_2.grid(column=0, row=inputs_row+2, sticky='ns')
        input_channel_3 = ttk.Label(main_frame, text='STOP2', font=self.normal_font)
        input_channel_3.grid(column=0, row=inputs_row+3, sticky='ns')

    def __num_of_frames(self, main_frame):

        # Number of frames in the data
        frame_label = ttk.Label(main_frame, text='Number of frames', font=self.normal_font)
        frame_label.grid(column=2, row=4, sticky='w')

        self.num_of_frames = IntVar(value=1)
        self.num_frames_entry = ttk.Entry(main_frame, textvariable=self.num_of_frames, width=3)
        self.num_frames_entry.grid(column=2, row=4, sticky='e')
        self.num_frames_entry.config(state='disabled')

        # Disable number of frames unless all inputs but one are empty
        self.input_start.trace('w', self.__check_if_empty)
        self.input_start.trace('w', self.__check_if_tag_lens_exists)
        self.input_stop1.trace('w', self.__check_if_empty)
        self.input_stop1.trace('w', self.__check_if_tag_lens_exists)
        self.input_stop2.trace('w', self.__check_if_empty)
        self.input_stop2.trace('w', self.__check_if_tag_lens_exists)

    def __outputs(self, main_frame):
        """ Wanted outputs """
        outputs_row = 6
        outputs_label = ttk.Label(main_frame, text='Outputs', font=self.bold_font)
        outputs_label.grid(column=4, row=outputs_row-1, sticky='ns')

        self.summed = BooleanVar(value=False)
        summed_array = ttk.Checkbutton(main_frame, text='Summed Stack', variable=self.summed)
        summed_array.grid(column=4, row=outputs_row, sticky='ns')
        self.memory = BooleanVar(value=False)
        in_memory = ttk.Checkbutton(main_frame, text='In Memory', variable=self.memory)
        in_memory.grid(column=4, row=outputs_row+1, sticky='ns')
        self.stack = BooleanVar(value=True)
        tif = ttk.Checkbutton(main_frame, text='Full Stack', variable=self.stack)
        tif.grid(column=4, row=outputs_row+2, sticky='ns')

    def __image_size(self, main_frame):
        # Define image sizes
        image_size_row = 1
        image_size_label = ttk.Label(main_frame, text='Image Size', font=self.bold_font)
        image_size_label.grid(column=2, row=image_size_row, sticky='ns', columnspan=1)
        x_size_label = ttk.Label(main_frame, text='X', font=self.normal_font)
        x_size_label.grid(column=2, row=image_size_row+1, sticky='w')
        y_size_label = ttk.Label(main_frame, text='Y', font=self.normal_font)
        y_size_label.grid(column=2, row=image_size_row+1, sticky='ns')
        z_size_label = ttk.Label(main_frame, text='Z', font=self.normal_font)
        z_size_label.grid(column=2, row=image_size_row+1, sticky='e')

        self.x_pixels = IntVar(value=512)
        self.y_pixels = IntVar(value=512)
        self.z_pixels = IntVar(value=1)

        x_pixels_entry = ttk.Entry(main_frame, textvariable=self.x_pixels, width=5)
        x_pixels_entry.grid(column=2, row=image_size_row+2, sticky='w')
        y_pixels_entry = ttk.Entry(main_frame, textvariable=self.y_pixels, width=5)
        y_pixels_entry.grid(column=2, row=image_size_row+2, sticky='ns')
        self.z_pixels_entry = ttk.Entry(main_frame, textvariable=self.z_pixels, width=5)
        self.z_pixels_entry.grid(column=2, row=image_size_row+2, sticky='e')
        self.z_pixels_entry.config(state='disabled')

    def __debug(self, main_frame):
        """ Read a smaller portion of data for debugging """
        debug_check = ttk.Checkbutton(main_frame, text='Debug?', variable=self.debug)
        debug_check.grid(column=2, row=7, sticky='ns')

    def __mirror_phase(self, main_frame):
        phase_text = ttk.Label(main_frame, text='Mirror phase [us]: ')
        phase_text.grid(column=0, row=1, sticky='w')
        phase_entry = ttk.Entry(main_frame, textvariable=self.phase, width=8)
        phase_entry.grid(column=0, row=1, sticky='e')

    def __reprate(self, main_frame):
        """ Laser repetition rate"""

        laser1_label = ttk.Label(main_frame, text='Laser rep. rate (FLIM) [Hz]')
        laser1_label.grid(column=2, row=6, sticky='ns')
        reprate_entry = ttk.Entry(main_frame, textvariable=self.reprate, width=11)
        reprate_entry.grid(column=3, row=6, sticky='ns')

    def __gating(self, main_frame):
        self.gating_check = ttk.Checkbutton(main_frame, text='With Gating?', variable=self.gating)
        self.gating_check.grid(column=2, row=5, sticky='ns')
        self.gating_check.config(state='disabled')

    def __binwidth(self, main_frame):
        """ Binwidth of Multiscaler (for FLIM) """

        binwidth_label = ttk.Label(main_frame, text='Multiscaler binwidth [sec]')
        binwidth_label.grid(column=2, row=1, sticky='ns')
        binwidth_entry = ttk.Entry(main_frame, textvariable=self.binwidth, width=9)
        binwidth_entry.grid(column=3, row=1, sticky='ns')

    def __tag_lens(self, main_frame):
        """ TAG lens nominal frequency """
        tag_row = 7
        tag_label = ttk.Label(main_frame, text='     TAG nominal freq. [Hz]\noffset [deg]                   n. pulses')
        tag_label.grid(column=0, row=tag_row, columnspan=2, sticky='w')
        tag_label_entry = ttk.Entry(main_frame, textvariable=self.tag_freq, width=10)
        tag_label_entry.grid(column=0, row=tag_row+1, sticky='ns')

        tag_pulses_entry = ttk.Entry(main_frame, textvariable=self.tag_pulses, width=3)
        tag_pulses_entry.grid(column=0, row=tag_row+1, sticky='e')
        tag_pulses_entry.config(state='disabled')

        self.tag_offset_entry = ttk.Entry(main_frame, textvariable=self.tag_offset, width=3)
        self.tag_offset_entry.grid(column=0, row=tag_row+1, sticky='w')
        self.tag_offset_entry.config(state='disabled')

    def __tag_bits(self, main_frame):
        """ TAG bits """
        tag_bits_row = 6
        tag_bits_label = ttk.Label(main_frame, text='TAG Bits Allocation', font=self.bold_font)
        tag_bits_label.grid(column=1, row=tag_bits_row, sticky='ns')

        self.tag_bits = BooleanVar(value=False)
        tag_bit_check = ttk.Checkbutton(main_frame, text='Use?', variable=self.tag_bits)
        tag_bit_check.grid(column=2, row=tag_bits_row, sticky='w')

        self.bits_grp_1_start = IntVar(value=1)
        self.bits_grp_1_end = IntVar(value=3)
        self.bits_grp_2_start = IntVar(value=4)
        self.bits_grp_2_end = IntVar(value=5)
        self.bits_grp_3_start = IntVar(value=6)
        self.bits_grp_3_end = IntVar(value=16)

        self.bits_grp_1_label = StringVar()
        self.bits_grp_2_label = StringVar()
        self.bits_grp_3_label = StringVar()

        self.tag_bits_group_options = ("Power", "Slow axis", "Fast axis", "Z axis", "None")

        bits_grp_1 = ttk.Combobox(main_frame, textvariable=self.bits_grp_1_label, width=10)
        bits_grp_1.grid(column=0, row=tag_bits_row+1, sticky='e')
        bits_grp_1.set('None')
        bits_grp_1['values'] = self.tag_bits_group_options

        bits_grp_2 = ttk.Combobox(main_frame, textvariable=self.bits_grp_2_label, width=10)
        bits_grp_2.grid(column=0, row=tag_bits_row+2, sticky='e')
        bits_grp_2.set('None')
        bits_grp_2['values'] = self.tag_bits_group_options

        bits_grp_3 = ttk.Combobox(main_frame, textvariable=self.bits_grp_3_label, width=10)
        bits_grp_3.grid(column=0, row=tag_bits_row+3, sticky='e')
        bits_grp_3.set('None')
        bits_grp_3['values'] = self.tag_bits_group_options

        bits_grp_1_start_lab = ttk.Label(main_frame, text='Start')
        bits_grp_1_start_lab.grid(column=1, row=tag_bits_row+1, sticky='w')
        bits_grp_1_start_ent = ttk.Entry(main_frame, textvariable=self.bits_grp_1_start, width=3)
        bits_grp_1_start_ent.grid(column=1, row=tag_bits_row+1, sticky='ns')
        bits_grp_1_end_lab = ttk.Label(main_frame, text='End')
        bits_grp_1_end_lab.grid(column=1, row=tag_bits_row+1, sticky='e')
        bits_grp_1_end_ent = ttk.Entry(main_frame, textvariable=self.bits_grp_1_end, width=3)
        bits_grp_1_end_ent.grid(column=2, row=tag_bits_row+1, sticky='w')

        bits_grp_2_start_lab = ttk.Label(main_frame, text='Start')
        bits_grp_2_start_lab.grid(column=1, row=tag_bits_row+2, sticky='w')
        bits_grp_2_start_ent = ttk.Entry(main_frame, textvariable=self.bits_grp_2_start, width=3)
        bits_grp_2_start_ent.grid(column=1, row=tag_bits_row+2, sticky='ns')
        bits_grp_2_end_lab = ttk.Label(main_frame, text='End')
        bits_grp_2_end_lab.grid(column=1, row=tag_bits_row+2, sticky='e')
        bits_grp_2_end_ent = ttk.Entry(main_frame, textvariable=self.bits_grp_2_end, width=3)
        bits_grp_2_end_ent.grid(column=2, row=tag_bits_row+2, sticky='w')

        bits_grp_3_start_lab = ttk.Label(main_frame, text='Start')
        bits_grp_3_start_lab.grid(column=1, row=tag_bits_row+3, sticky='w')
        bits_grp_3_start_ent = ttk.Entry(main_frame, textvariable=self.bits_grp_3_start, width=3)
        bits_grp_3_start_ent.grid(column=1, row=tag_bits_row+3, sticky='ns')
        bits_grp_3_end_lab = ttk.Label(main_frame, text='End')
        bits_grp_3_end_lab.grid(column=1, row=tag_bits_row+3, sticky='e')
        bits_grp_3_end_ent = ttk.Entry(main_frame, textvariable=self.bits_grp_3_end, width=3)
        bits_grp_3_end_ent.grid(column=2, row=tag_bits_row+3, sticky='w')

        self.tag_bits_dict = {}
        self.tag_bits_dict = {0: TagBits(value=self.bits_grp_1_label.get(),
                                         start=self.bits_grp_1_start.get(),
                                         end=self.bits_grp_1_end.get()),
                              1: TagBits(value=self.bits_grp_2_label.get(),
                                         start=self.bits_grp_2_start.get(),
                                         end=self.bits_grp_2_end.get()),
                              2: TagBits(value=self.bits_grp_3_label.get(),
                                         start=self.bits_grp_3_start.get(),
                                         end=self.bits_grp_3_end.get())}

    def __fill_frac(self, main_frame):
        """ Percentage of time mirrors spend "inside" the image """

        fill_frac_text = ttk.Label(main_frame, text='Fill fraction [%]: ')
        fill_frac_text.grid(column=0, row=4, sticky='w')
        fill_frac_entry = ttk.Entry(main_frame, textvariable=self.fill_frac, width=8)
        fill_frac_entry.grid(column=0, row=4, sticky='e')

    def __browsefunc(self):
        filetypes = [('List files', '*.lst'), ('Pickle files', '*.p')]
        if self.filename.get() != '':
            self.filename.set(filedialog.askopenfilename(filetypes=filetypes, title='Choose a list\pickle file',
                                                     initialdir=str(Path(self.filename.get()).parent)))
        else:
            self.filename.set(filedialog.askopenfilename(filetypes=filetypes, title='Choose a list\pickle file',
                                                     initialdir='.'))

    def __check_if_empty(self, *args):
        list_of_values = [self.input_start.get(), self.input_stop1.get(), self.input_stop2.get()]
        if 2 == list_of_values.count('Empty'):
            if 'PMT1' in list_of_values or 'PMT2' in list_of_values:
                self.num_frames_entry.config(state='normal')
            else:
                self.num_frames_entry.config(state='disabled')

    def __check_if_tag_lens_exists(self, *args):
        list_of_values = [self.input_start.get(), self.input_stop1.get(), self.input_stop2.get()]
        if 'TAG Lens' in list_of_values:
            self.z_pixels_entry.config(state='normal')
            # self.tag_offset_entry.config(state='normal')
        else:
            self.z_pixels_entry.config(state='disabled')
            # self.tag_offset_entry.config(state='disabled')

    def __bidir(self, main_frame):
        """ Checkbox for bi-directional scan """

        bidir_check = ttk.Checkbutton(main_frame, text='Bi-directional scan', variable=self.bidir)
        bidir_check.grid(column=0, row=5, sticky='ns')
        self.bidir.trace('w', self.__check_if_bidir)

    def __check_if_bidir(self, *args):
        if self.bidir:
            self.keep_unidir_check.config(state='normal')
        if not self.bidir:
            self.keep_unidir_check.config(state='disabled')

    def __keep_unidir_events(self, main_frame):
        """ Checkbox to see if events taken in the returning phase of a resonant mirror should be kept. """
        self.keep_unidir_check = ttk.Checkbutton(main_frame, text='Keep unidirectional?',
                                                 variable=self.keep_unidir)
        self.keep_unidir_check.grid(column=0, row=6, sticky='ns')
        self.keep_unidir_check.config(state='disabled')

    def __flim(self, main_frame):
        """
        Defines the mapping between one pulse and the missing pulses.
        For example, downsampling factor of 8 means that every pulse that is
        received starts an event of 8 pulses, with the next recorded pulse being the 9th.
        :param main_frame: ttk.Frame
        """
        flim_check: ttk.Checkbutton = ttk.Checkbutton(main_frame,
                                                      variable=self.flim,
                                                      text='FLIM?')
        flim_check.grid(row=2, column=2, sticky='ns')
        self.flim.trace('w', self.__check_if_flim)

    def __censor(self, main_frame):
        """
        If FLIM is active, this checkbox enables the use of censor correction on the generated images.
        :param main_frame: ttk.Frame
        """
        self.censor_check: ttk.Checkbutton = ttk.Checkbutton(main_frame, variable=self.censor,
                                                             text='Censor Correction')
        self.censor_check.grid(row=3, column=2, sticky='ns')
        self.censor_check.config(state='disabled')

    def __check_if_flim(self, *args):
        if self.flim:
            self.censor_check.config(state='normal')
            self.gating_check.config(state='normal')
        else:
            self.censor_check.config(state='disabled')
            self.gating_check.config(state='disabled')
        self.root.update_idletasks()

    def __line_freq(self, main_frame):
        """ Frequency of the line scanning mirror """
        line_freq_label = ttk.Label(main_frame, text="Line freq [Hz]: ")
        line_freq_label.grid(row=3, column=0, sticky='w')
        line_freq_entry = ttk.Entry(main_frame, textvariable=self.line_freq, width=8)
        line_freq_entry.grid(row=3, column=0, sticky='e')

    def __sweeps_as_lines(self, main_frame):
        """ Use the sweeps as lines for the image generation """
        sweeps_cb = ttk.Checkbutton(main_frame, variable=self.sweeps_as_lines,
                                      text='Sweeps as lines?')
        sweeps_cb.grid(row=4, column=2, sticky='ns')

    def __advanced_win(self, main_frame):
        advanced_but = ttk.Button(main_frame, text="Advanced", command=self.__open_advanced)
        advanced_but.grid(row=10, column=4, sticky='ns')

    def __open_advanced(self, *args):
        self.advanced_win = Toplevel(self.root)
        frame = ttk.Frame(self.advanced_win, width=300, height=300)
        frame.grid(column=0, row=0)
        frame['borderwidth'] = 2
        style = ttk.Style()
        style.theme_use('clam')
        self.__setup_advanced_frame(frame)
        self.__gating(frame)
        self.__flim(frame)
        self.__censor(frame)
        self.__sweeps_as_lines(frame)
        self.__debug(frame)
        self.__mirror_phase(frame)
        self.__fill_frac(frame)
        self.__reprate(frame)
        self.__binwidth(frame)
        self.__keep_unidir_events(frame)
        self.__bidir(frame)
        self.__check_if_bidir(frame)
        self.__tag_lens(frame)
        self.__frame_delay(frame)
        self.__line_freq(frame)
        for child in frame.winfo_children():
            child.grid_configure(padx=3, pady=2)

    def __setup_advanced_frame(self, frame):
        scan_lab = ttk.Label(frame, text='       Scanner Settings', font=self.bold_font)
        scan_lab.grid(row=0, column=0, sticky='ns')

        hardware_lab = ttk.Label(frame, text='               Hardware Settings', font=self.bold_font)
        hardware_lab.grid(row=0, column=2, sticky='ns')

    def __frame_delay(self, main_frame):
        frame_delay_label = ttk.Label(main_frame, text="Frame delay [sec]: ")
        frame_delay_label.grid(row=2, column=0, sticky='w')
        frame_delay_entry = ttk.Entry(main_frame, textvariable=self.frame_delay, width=8)
        frame_delay_entry.grid(row=2, column=0, sticky='e')

    ####### ONLY SAVE\LOAD FUNCS AFTER THIS POINT ########

    def __save_cfg(self, main_frame):
        """ A button to write a .json with current configs """
        config_label = ttk.Label(main_frame, text='Configuration File', font=self.bold_font)
        config_label.grid(column=1, row=self.config_row, sticky='ns')
        self.save_as: StringVar = StringVar(value='default')
        save_label = ttk.Label(main_frame, text='Config file name to save:')
        save_label.grid(column=0, row=self.config_row+1, sticky='ns', columnspan=2, padx=10)
        save_entry = ttk.Entry(main_frame, textvariable=self.save_as, width=8)
        save_entry.grid(column=1, row=self.config_row+1, sticky='e')
        save_button = ttk.Button(main_frame, text="Save cfg", command=self.__callback_save_cur_cfg)
        save_button.grid(column=1, row=self.config_row+2, sticky='w')

    def __callback_save_cur_cfg(self) -> None:
        """
        Takes a GUIApp() instance and saves it to a .json file
        """
        cfg_dict_to_save: Dict[str, Tuple[str]] = {}
        for key, val in self.__dict__.items():
            if key.find('entry') == -1 \
                and key.find('root') == -1 \
                and key.find('check') == -1 \
                and key.find('cfg') == -1 \
                and key.find('dict') == -1 \
                and key.find('config') == -1:
                try:
                    data_to_save = (str(val), val.get())
                    cfg_dict_to_save[key] = data_to_save
                except AttributeError:
                    pass  # don't save non-tkinter variables
                except TypeError:  # The TAG bits variable
                    # warnings.warn(f"Error with value {val} under key {key}. Configuration file not saved. ")
                    cfg_dict_to_save[key] = val

        path_to_save_to: str = str(Path(__file__).parent / 'configs') + sep + str(self.save_as.get()) + '.json'
        with open(path_to_save_to, 'w') as f:
            json.dump(cfg_dict_to_save, f, indent=4)

    def __load_cfg(self, main_frame: ttk.Frame):
        """
        Load a specific .json file and change all variables accordingly
        """
        self.cfg_filename: StringVar = StringVar(value='default')
        load_button: Button = ttk.Button(main_frame, text="Load cfg", command=self.__browsecfg)
        load_button.grid(column=1, row=self.config_row+2, sticky='e')

    def __browsecfg(self, new_cfg=None):
        if not new_cfg:
            self.cfg_filename.set(filedialog.askopenfilename(filetypes=[('Config files', '*.json')],
                                                             title='Choose a configuration file',
                                                             initialdir=str(Path(__file__).parent / 'configs')))
        else:
            self.cfg_filename.set(new_cfg)
        with open(self.cfg_filename.get(), 'r') as f:
            self.config = json.load(f)
            utime(self.cfg_filename.get(), (time.time(), time.time()))
        self.__modify_vars()

    def __modify_vars(self):
        """
        With the dictionary loaded from the .json file, change all variables
        """
        for key, val in self.config.items():
            try:
                self.x_pixels._tk.globalsetvar(val[0], val[1])
            except:
                pass
        self.root.update_idletasks()

    def __get_matching_tkinter_var(self, val: Union[StringVar, DoubleVar, IntVar]) -> \
        Union[StringVar, DoubleVar, IntVar, BooleanVar]:
        """
        Create a tkinter variable (StringVar(), for example) that will
        be set after reading a config file.
        :param val: Value to be set
        :return: A Tkinter variable object.
        """
        if type(val) == str:
            return StringVar(value=val)
        elif type(val) == int:
            return IntVar(value=val)
        elif type(val) == float:
            return DoubleVar(value=val)
        elif type(val) == bool:
            return BooleanVar(value=val)
        else:
            raise ValueError(f'Type not recognized for value {val}.')

    def __load_last_used_cfg(self, main_frame):
        dir: WindowsPath = Path(__file__).parent / 'configs'
        all_cfg_files: Iterable = dir.glob('*.json')
        latest_filename: str = ''
        latest_file_date: int = 0
        for cfg_file in all_cfg_files:
            cur_date_modified = cfg_file.stat()[8]
            if cur_date_modified > latest_file_date:
                latest_filename = str(cfg_file)
                latest_file_date = cur_date_modified

        if latest_filename != '':
            with open(latest_filename, 'r') as f:
                try:
                    self.config = json.load(f)
                except json.decoder.JSONDecodeError:  # Unable to load .json config
                    self.config = {}
            self.__modify_vars()


def verify_gui_input(gui):
    """Validate all GUI inputs"""
    data_sources = set(gui.tuple_of_data_sources)
    channel_inputs = {gui.input_start, gui.input_stop1, gui.input_stop2}
    MINIMAL_TAG_BIT = 1
    MAXIMAL_TAG_BIT = 16

    if gui.input_start != 'PMT1':
        if gui.input_stop1 != 'PMT1':
            if gui.input_stop2 != 'PMT1':
                raise BrokenPipeError('PMT1 value has to be entered in inputs.')

    if gui.num_of_frames == None:
        if 'Frames' not in data_sources:
            raise BrokenPipeError('You must either enter a frame channel or number of frames.')
    else:
        if float(gui.num_of_frames) != gui.num_of_frames:
            raise ValueError('Please enter an integer number of frames.')

    if gui.num_of_frames < 0:
        raise ValueError('Number of frames has to be a positive number.')

    filename = gui.filename
    if not filename.endswith('.lst') and not filename.endswith('.p'):
        raise BrokenPipeError('Please choose a list (*.lst) or pickle (*.p) file for analysis.')

    if channel_inputs > data_sources:
        raise ValueError('Wrong inputs in channels. Please choose a value from the list.')

    list_of_keys = [gui.input_start, gui.input_stop1, gui.input_stop2]
    set_of_keys = set(list_of_keys)

    if len(list_of_keys) != len(set_of_keys):  # making sure only a single option was chosen in the GUI
        if [x for x in list_of_keys if x != 'Empty'] != list(set_of_keys.difference({'Empty'})):
            raise KeyError('Input consisted of two or more similar names which are not "Empty".')

    # TAG bits input verification
    if gui.tag_bits:
        values_of_bits_set = set()
        start_bits_set = set()
        end_bits_set = set()
        for key, val in gui.tag_bits_dict.items():
            if val.value not in gui.tag_bits_group_options:
                raise UserWarning(f"Value {val} not in allowed TAG bits inputs.")
            if not isinstance(val.start, int):
                raise UserWarning(f"The start bit of TAG label {val.value} wasn't an integer.")
            if not isinstance(val.end, int):
                raise UserWarning(f"The end bit of TAG label {val.value} wasn't an integer.")
            if val.end < val.start:
                raise UserWarning(f"Bits in row {key + 1} have a start value which is higher than its end.")
            if val.start > MAXIMAL_TAG_BIT or val.end > MAXIMAL_TAG_BIT:
                raise UserWarning(f"In label {key} maximal allowed TAG bit is {MAXIMAL_TAG_BIT}.")
            if val.start < MINIMAL_TAG_BIT or val.end < MINIMAL_TAG_BIT:
                raise UserWarning(f"In label {key} minimal allowed TAG bit is {MINIMAL_TAG_BIT}.")
            values_of_bits_set.add(val)
            start_bits_set.add(val.start)
            end_bits_set.add(val.end)

        if len(values_of_bits_set) > len(start_bits_set):
            raise UserWarning("Some TAG bit labels weren't given unique start bits.")

        if len(values_of_bits_set) > len(end_bits_set):
            raise UserWarning("Some TAG bit labels weren't given unique end bits.")

    if not isinstance(gui.phase, float) and not isinstance(gui.phase, int):
        raise UserWarning('Mirror phase must be a number.')

    if gui.fill_frac < 0:
        raise UserWarning('Fill fraction must be a positive number.')

    if not isinstance(gui.fill_frac, float) and not isinstance(gui.flyback, int):
        raise UserWarning('Fill fraction must be a number.')

    if gui.x_pixels < 0:
        raise UserWarning('X pixels value must be greater than 0.')

    if gui.y_pixels < 0:
        raise UserWarning('X pixels value must be greater than 0.')

    if gui.z_pixels < 0:
        raise UserWarning('X pixels value must be greater than 0.')
    try:
        int(gui.x_pixels)
        int(gui.x_pixels)
        int(gui.x_pixels)
    except ValueError:
        raise UserWarning('Pixels must be an integer number.')

    if float(gui.x_pixels) != gui.x_pixels:
        raise UserWarning('Enter an integer number for the x-axis pixels.')

    if float(gui.y_pixels) != gui.y_pixels:
        raise UserWarning('Enter an integer number for the y-axis pixels.')

    if float(gui.z_pixels) != gui.z_pixels:
        raise UserWarning('Enter an integer number for the z-axis pixels.')

    if gui.reprate < 0:
        raise UserWarning('Laser repetition rate must be positive.')

    if gui.binwidth < 0:
        raise UserWarning('Binwidth must be a positive number.')

    if gui.binwidth > 1e-9:
        raise UserWarning('Enter a binwidth with units of [seconds].')

    if type(gui.filename) != str:
        raise UserWarning('Filename must be a string.')

    if 'Laser' in channel_inputs and gui.flim == 1:
        raise UserWarning("Can't have both a laser channel active and the FLIM checkboxed ticked.")

    if not gui.imaging_software.upper() in [name for name, member in ImagingSoftware.__members__.items()]:
        raise UserWarning("Must use existing options in the Imaging Software entry.")

    if not isinstance(gui.frame_delay, float):
        raise UserWarning("Frame delay must be a float.")

    if gui.frame_delay < 0:
        raise UserWarning("Frame delay must be a positive float.")

    if gui.frame_delay > 10:
        raise UserWarning("Frame delay is the number of seconds between subsequent frames.")


if __name__ == '__main__':
    app = GUIAppLst()
