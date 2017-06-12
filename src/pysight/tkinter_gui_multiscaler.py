"""
__author__ = Hagai Hargil
"""
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import json
from typing import Dict, Union, Tuple, Iterable
from pathlib import Path, WindowsPath
from os import sep, utime
import time


class GUIApp(object):
    """
    Main GUI for the multiscaler code.
    Note - class variables should contain "entry" in their name if they point
    to an entry TTK object. Also, no variable should contain "root" in its name.
    """
    def __init__(self):
        self.root = Tk()
        self.root.title("Multiscaler Readout and Display")
        self.root.rowconfigure(5, weight=1)
        self.root.columnconfigure(5, weight=1)
        main_frame = ttk.Frame(self.root, width=800, height=800)
        main_frame.grid(column=0, row=0)
        main_frame['borderwidth'] = 2
        style = ttk.Style()
        style.theme_use('clam')

        # Run widgets
        self.__browse_file(main_frame)
        self.__input_channels(main_frame)
        self.__num_of_frames(main_frame)
        self.__outputs(main_frame)
        self.__image_size(main_frame)
        self.__debug(main_frame)
        self.__mirror_phase(main_frame)
        self.__fill_frac(main_frame)
        self.__reprate(main_frame)
        self.__binwidth(main_frame)
        self.__tag_lens(main_frame)
        self.__tag_bits(main_frame)
        self.__bi_dir(main_frame)
        self.__keep_unidir_events(main_frame)
        self.__save_cfg(main_frame)
        self.__load_cfg(main_frame)
        self.__load_last_used_cfg(main_frame)

        # Define the last quit button and wrap up GUI
        quit_button = ttk.Button(self.root, text='Start', command=self.root.destroy)
        quit_button.grid()

        for child in main_frame.winfo_children():
            child.grid_configure(padx=3, pady=2)
        self.root.wait_window()

    def __browse_file(self, main_frame):
        self.filename = StringVar(value="")

        browse_button = ttk.Button(main_frame, text="Browse", command=self.__browsefunc)
        browse_button.grid(column=0, row=0, sticky='ns')

    def __input_channels(self, main_frame):
        # Comboboxes
        input_channels_label = ttk.Label(main_frame, text='Input Channels')
        input_channels_label.grid(columns=3, row=0, sticky='se')
        self.input_start = StringVar()
        self.input_stop1 = StringVar()
        self.input_stop2 = StringVar()
        self.tuple_of_data_sources = ('PMT1', 'Lines', 'Frames', 'Laser', 'TAG Lens', 'Empty')  # TODO: No PMT2 currently
        mb1 = ttk.Combobox(main_frame, textvariable=self.input_start, width=10)
        mb1.grid(column=2, row=1, sticky='w')
        mb1.set('Frames')
        mb1['values'] = self.tuple_of_data_sources
        mb2 = ttk.Combobox(main_frame, textvariable=self.input_stop1, width=10)
        mb2.grid(column=2, row=2, sticky='w')
        mb2.set('PMT1')
        mb2['values'] = self.tuple_of_data_sources
        mb3 = ttk.Combobox(main_frame, textvariable=self.input_stop2, width=10)
        mb3.grid(column=2, row=3, sticky='w')
        mb3.set('Lines')
        mb3['values'] = self.tuple_of_data_sources

        # Labels
        input_channel_1 = ttk.Label(main_frame, text='START')
        input_channel_1.grid(column=1, row=1, sticky='ns')
        input_channel_2 = ttk.Label(main_frame, text='STOP1')
        input_channel_2.grid(column=1, row=2, sticky='ns')
        input_channel_3 = ttk.Label(main_frame, text='STOP2')
        input_channel_3.grid(column=1, row=3, sticky='ns')

    def __num_of_frames(self, main_frame):

        # Number of frames in the data
        frame_label = ttk.Label(main_frame, text='Number of frames')
        frame_label.grid(column=0, row=1, sticky='ns')

        self.num_of_frames = IntVar(value=1)
        self.num_frames_entry = ttk.Entry(main_frame, textvariable=self.num_of_frames, width=3)
        self.num_frames_entry.grid(column=0, row=2, sticky='ns')
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
        outputs_label = ttk.Label(main_frame, text='Outputs:')
        outputs_label.grid(column=0, row=3, sticky='w')

        self.summed = IntVar()
        summed_array = ttk.Checkbutton(main_frame, text='Summed array', variable=self.summed)
        summed_array.grid(column=0, row=4, sticky='w')
        self.full = IntVar()
        full_array = ttk.Checkbutton(main_frame, text='Full array', variable=self.full)
        full_array.grid(column=1, row=4, sticky='ns')
        self.tif = IntVar(value=1)
        tif = ttk.Checkbutton(main_frame, text='Tiff', variable=self.tif)
        tif.grid(column=2, row=4, sticky='ns')

    def __image_size(self, main_frame):

        # Define image sizes
        image_size_label = ttk.Label(main_frame, text='Image Size')
        image_size_label.grid(column=6, row=0, sticky='ns')
        x_size_label = ttk.Label(main_frame, text='X')
        x_size_label.grid(column=6, row=1, sticky='w')
        y_size_label = ttk.Label(main_frame, text='Y')
        y_size_label.grid(column=6, row=1, sticky='ns')
        z_size_label = ttk.Label(main_frame, text='Z')
        z_size_label.grid(column=6, row=1, sticky='e')

        self.x_pixels = IntVar(value=512)
        self.y_pixels = IntVar(value=512)
        self.z_pixels = IntVar(value=100)

        x_pixels_entry = ttk.Entry(main_frame, textvariable=self.x_pixels, width=5)
        x_pixels_entry.grid(column=6, row=2, sticky='w')
        y_pixels_entry = ttk.Entry(main_frame, textvariable=self.y_pixels, width=5)
        y_pixels_entry.grid(column=6, row=2, sticky='ns')
        self.z_pixels_entry = ttk.Entry(main_frame, textvariable=self.z_pixels, width=5)
        self.z_pixels_entry.grid(column=6, row=2, sticky='e')
        self.z_pixels_entry.config(state='disabled')

    def __debug(self, main_frame):
        """ Read a smaller portion of data for debugging """
        self.debug = IntVar()
        debug_check = ttk.Checkbutton(main_frame, text='Debug?', variable=self.debug)
        debug_check.grid(column=6, row=11, sticky='ns')

    def __mirror_phase(self, main_frame):
        self.phase = DoubleVar(value=-2.6)
        phase_text = ttk.Label(main_frame, text='Mirror phase [rad]: ')
        phase_text.grid(column=6, row=5, sticky='w')
        phase_entry = ttk.Entry(main_frame, textvariable=self.phase, width=5)
        phase_entry.grid(column=6, row=5, sticky='e')

    def __reprate(self, main_frame):
        """ Laser repetition rate"""

        laser1_label = ttk.Label(main_frame, text='Laser rep. rate (FLIM) [Hz]')
        laser1_label.grid(column=0, row=9, sticky='w')

        self.reprate = DoubleVar(value=80e6)  # 80e6 for the Chameleon, 0 to raise ZeroDivisionError
        reprate_entry = ttk.Entry(main_frame, textvariable=self.reprate, width=11)
        reprate_entry.grid(column=1, row=9, sticky='w')

        self.offset = DoubleVar(value=3.5)  # difference between pulse and arrival to sample
        laser2_label = ttk.Label(main_frame, text='Offset [ns]')
        laser2_label.grid(column=2, row=9, sticky='w')
        offset_entry = ttk.Entry(main_frame, textvariable=self.offset, width=3)
        offset_entry.grid(column=2, row=9, sticky='e')

    def __binwidth(self, main_frame):
        """ Binwidth of Multiscaler (for FLIM) """

        binwidth_label = ttk.Label(main_frame, text='Binwidth of Multiscaler [sec]')
        binwidth_label.grid(column=0, row=10, sticky='ns')
        self.binwidth = DoubleVar(value=800e-12)
        binwidth_entry = ttk.Entry(main_frame, textvariable=self.binwidth, width=10)
        binwidth_entry.grid(column=1, row=10, sticky='ns')

    def __tag_lens(self, main_frame):
        """ TAG lens nominal frequency """

        tag_label = ttk.Label(main_frame, text='TAG nominal frequency [Hz]\nand number of pulses')
        tag_label.grid(column=6, row=8, sticky='ns')
        self.tag_freq = StringVar(value=0.1898e6)
        tag_label_entry = ttk.Entry(main_frame, textvariable=self.tag_freq, width=10)
        tag_label_entry.grid(column=6, row=9, sticky='ns')

        self.tag_pulses = IntVar(value=1)
        tag_pulses_entry = ttk.Entry(main_frame, textvariable=self.tag_pulses, width=3)
        tag_pulses_entry.grid(column=6, row=9, sticky='e')
        tag_pulses_entry.config(state='disabled')

    def __tag_bits(self, main_frame):
        """ TAG bits """

        tag_bits_label = ttk.Label(main_frame, text='TAG Bits Allocation')
        tag_bits_label.grid(column=1, row=5, sticky='ns')

        self.tag_bits = IntVar(value=0)
        tag_bit_check = ttk.Checkbutton(main_frame, text='Use?', variable=self.tag_bits)
        tag_bit_check.grid(column=2, row=5, sticky='ns')

        slow_axis_label = ttk.Label(main_frame, text='Slow Axis:')
        slow_axis_label.grid(column=0, row=6, sticky='e')
        fast_axis_label = ttk.Label(main_frame, text='Fast Axis:')
        fast_axis_label.grid(column=0, row=7, sticky='e')
        z_axis_label = ttk.Label(main_frame, text='Z Axis:')
        z_axis_label.grid(column=0, row=8, sticky='e')

        self.slow_bit_start = IntVar(value=1)
        self.slow_bit_end = IntVar(value=3)
        self.fast_bit_start = IntVar(value=4)
        self.fast_bit_end = IntVar(value=5)
        self.z_bit_start = IntVar(value=6)
        self.z_bit_end = IntVar(value=16)

        slow_start = ttk.Label(main_frame, text='Start')
        slow_start.grid(column=1, row=6, sticky='w')
        slow_start_ent = ttk.Entry(main_frame, textvariable=self.slow_bit_start, width=3)
        slow_start_ent.grid(column=1, row=6, sticky='ns')
        slow_end = ttk.Label(main_frame, text='End')
        slow_end.grid(column=1, row=6, sticky='e')
        slow_end_ent = ttk.Entry(main_frame, textvariable=self.slow_bit_end, width=3)
        slow_end_ent.grid(column=2, row=6, sticky='w')

        fast_start = ttk.Label(main_frame, text='Start')
        fast_start.grid(column=1, row=7, sticky='w')
        fast_start_ent = ttk.Entry(main_frame, textvariable=self.fast_bit_start, width=3)
        fast_start_ent.grid(column=1, row=7, sticky='ns')
        fast_end = ttk.Label(main_frame, text='End')
        fast_end.grid(column=1, row=7, sticky='e')
        fast_end_ent = ttk.Entry(main_frame, textvariable=self.fast_bit_end, width=3)
        fast_end_ent.grid(column=2, row=7, sticky='w')

        z_start = ttk.Label(main_frame, text='Start')
        z_start.grid(column=1, row=8, sticky='w')
        z_start_ent = ttk.Entry(main_frame, textvariable=self.z_bit_start, width=3)
        z_start_ent.grid(column=1, row=8, sticky='ns')
        z_end = ttk.Label(main_frame, text='End')
        z_end.grid(column=1, row=8, sticky='e')
        z_end_ent = ttk.Entry(main_frame, textvariable=self.z_bit_end, width=3)
        z_end_ent.grid(column=2, row=8, sticky='w')

    def __fill_frac(self, main_frame):
        """ Percentage of time mirrors spend "inside" the image """

        self.fill_frac = DoubleVar(value=80)  # percent
        fill_frac_text = ttk.Label(main_frame, text='Fill fraction [%]: ')
        fill_frac_text.grid(column=6, row=6, sticky='w')
        fill_frac_entry = ttk.Entry(main_frame, textvariable=self.fill_frac, width=4)
        fill_frac_entry.grid(column=6, row=6, sticky='e')

    def __browsefunc(self):
        if self.filename.get() != '':
            self.filename.set(filedialog.askopenfilename(filetypes=[('List files', '*.lst')], title='Choose a list file',
                                                     initialdir=str(Path(self.filename.get()).parent)))
        else:
            self.filename.set(filedialog.askopenfilename(filetypes=[('List files', '*.lst')], title='Choose a list file',
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
        else:
            self.z_pixels_entry.config(state='disabled')

    def __bi_dir(self, main_frame):
        """ Checkbox for bi-directional scan """

        self.bidir = IntVar(value=1)
        bidir_check = ttk.Checkbutton(main_frame, text='Bi-directional scan', variable=self.bidir)
        bidir_check.grid(column=6, row=3, sticky='ns')
        self.bidir.trace('w', self.__check_if_bidir)

    def __check_if_bidir(self, *args):
        if self.bidir:
            self.keep_unidir_check.config(state='normal')

    def __keep_unidir_events(self, main_frame):
        """ Checkbox to see if events taken in the returning phase of a resonant mirror should be kept. """
        self.keep_unidir = IntVar(value=0)
        self.keep_unidir_check = ttk.Checkbutton(main_frame, text='Keep unidirectional?', variable=self.keep_unidir)
        self.keep_unidir_check.grid(column=6, row=4, sticky='ns')
        self.keep_unidir_check.config(state='disabled')

    def __save_cfg(self, main_frame):
        """ A button to write a .json with current configs """
        self.cfg_to_save: StringVar = StringVar(value='default')
        save_label = ttk.Label(main_frame, text='Config file name to save:')
        save_label.grid(column=0, row=11, sticky='w')
        save_entry = ttk.Entry(main_frame, textvariable=self.cfg_to_save, width=10)
        save_entry.grid(column=1, row=11, sticky='w')
        save_button = ttk.Button(main_frame, text="Save cfg", command=self.__callback_save_cur_cfg)
        save_button.grid(column=2, row=11, sticky='w')

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
                and key.find('config') == -1:
                try:
                    data_to_save = (str(val), val.get())
                    cfg_dict_to_save[key] = data_to_save
                except AttributeError:
                    pass  # don't save non-tkinter variables

        path_to_save_to: str = str(Path(__file__).parent / 'configs') + sep + str(self.cfg_to_save.get()) + '.json'
        with open(path_to_save_to, 'w') as f:
            json.dump(cfg_dict_to_save, f, indent=4)

    def __load_cfg(self, main_frame: ttk.Frame):
        """
        Load a specific .json file and change all variables accordingly
        """
        self.cfg_filename: StringVar = StringVar(value='default')
        load_button: Button = ttk.Button(main_frame, text="Load cfg", command=self.__browsecfg)
        load_button.grid(column=4, row=11, sticky='w')

    def __browsecfg(self):
        self.cfg_filename.set(filedialog.askopenfilename(filetypes=[('Config files', '*.json')],
                                                         title='Choose a configuration file',
                                                         initialdir=str(Path(__file__).parent / 'configs')))
        with open(self.cfg_filename.get(), 'r') as f:
            self.config = json.load(f)
            utime(self.cfg_filename.get(), (time.time(), time.time()))
        self.__modify_vars()

    def __modify_vars(self):
        """
        With the dictionary loaded from the .json file, change all variables
        """
        for key, val in self.config.items():
            # val_to_set = self.__get_matching_tkinter_var(val)
            self.x_pixels._tk.globalsetvar(val[0], val[1])
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
            raise ValueError('Type not recognized for value {}.'.format(val))

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
                self.config = json.load(f)
            self.__modify_vars()


def verify_gui_input(gui):
    """Validate all GUI inputs"""
    data_sources = set(gui.tuple_of_data_sources)
    channel_inputs = {gui.input_start.get(), gui.input_stop1.get(), gui.input_stop2.get()}

    if gui.input_start.get() != 'PMT1':
        if gui.input_stop1.get() != 'PMT1':
            if gui.input_stop2.get() != 'PMT1':
                raise BrokenPipeError('PMT1 value has to be entered in inputs.')

    if gui.num_of_frames.get() == None:
        if 'Frames' not in data_sources:
            raise BrokenPipeError('You must either enter a frame channel or number of frames.')
    else:
        if float(gui.num_of_frames.get()) != gui.num_of_frames.get():
            raise ValueError('Please enter an integer number of frames.')

    if gui.num_of_frames.get() < 0:
        raise ValueError('Number of frames has to be a positive number.')

    filename = gui.filename.get()
    if not filename.endswith('.lst'):
        raise BrokenPipeError('Please choose a list (*.lst) file for analysis.')

    if channel_inputs > data_sources:
        raise ValueError('Wrong inputs in channels. Please choose a value from the list.')

    list_of_keys = [gui.input_start.get(), gui.input_stop1.get(), gui.input_stop2.get()]
    set_of_keys = set(list_of_keys)

    if len(list_of_keys) != len(set_of_keys):  # making sure only a single option was chosen in the GUI
        if [x for x in list_of_keys if x != 'Empty'] != list(set_of_keys.difference({'Empty'})):
            raise KeyError('Input consisted of two or more similar names which are not "Empty".')

    # TAG bits input verification
    set_of_tags = {gui.slow_bit_start.get(), gui.slow_bit_end.get(),
                   gui.fast_bit_start.get(), gui.fast_bit_end.get(),
                   gui.z_bit_start.get(), gui.z_bit_end.get()}

    for num in set_of_tags:
        assert isinstance(num, int), 'TAG bit has to be an integer.'

    if len(set_of_tags) != 6:
        raise UserWarning('Conflicting starts and ends of TAG bits. Take note that bits are inclusive on both ends.')

    if max(set_of_tags) > 16:
        raise UserWarning('Maximal TAG bit is 16.')

    if min(set_of_tags) < 1:
        raise UserWarning('Minimal TAG bit is 1.')

    if gui.slow_bit_start.get() > gui.slow_bit_end.get():
        raise UserWarning('Slow bit end is smaller than its start.')

    if gui.fast_bit_start.get() > gui.fast_bit_end.get():
        raise UserWarning('Fast bit end is smaller than its start.')

    if gui.z_bit_start.get() > gui.z_bit_end.get():
        raise UserWarning('Z bit end is smaller than its start.')

    if 0 == gui.summed and 0 == gui.tif and 0 == gui.full:
        raise UserWarning('No outputs chosen. Please check at least one.')

    if not isinstance(gui.phase.get(), float) and not isinstance(gui.phase.get(), int):
        raise UserWarning('Mirror phase must be a number.')

    if gui.fill_frac.get() < 0:
        raise UserWarning('Fill fraction must be a positive number.')

    if not isinstance(gui.fill_frac.get(), float) and not isinstance(gui.flyback.get(), int):
        raise UserWarning('Fill fraction must be a number.')

    if gui.x_pixels.get() < 0:
        raise UserWarning('X pixels value must be greater than 0.')

    if gui.y_pixels.get() < 0:
        raise UserWarning('X pixels value must be greater than 0.')

    if gui.z_pixels.get() < 0:
        raise UserWarning('X pixels value must be greater than 0.')
    try:
        int(gui.x_pixels.get())
        int(gui.x_pixels.get())
        int(gui.x_pixels.get())
    except ValueError:
        raise UserWarning('Pixels must be an integer number.')

    if float(gui.x_pixels.get()) != gui.x_pixels.get():
        raise UserWarning('Enter an integer number for the x-axis pixels.')

    if float(gui.y_pixels.get()) != gui.y_pixels.get():
        raise UserWarning('Enter an integer number for the y-axis pixels.')

    if float(gui.z_pixels.get()) != gui.z_pixels.get():
        raise UserWarning('Enter an integer number for the z-axis pixels.')

    if gui.reprate.get() < 0:
        raise UserWarning('Laser repetition rate must be positive.')

    if gui.binwidth.get() < 0:
        raise UserWarning('Binwidth must be a positive number.')

    if gui.binwidth.get() > 1e-9:
        raise UserWarning('Enter a binwidth with units of [seconds].')

    if type(gui.filename.get()) != str:
        raise UserWarning('Filename must be a string.')


if __name__ == '__main__':
    app = GUIApp()
