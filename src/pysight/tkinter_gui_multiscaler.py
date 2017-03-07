"""
__author__ = Hagai Hargil
"""
from tkinter import *
from tkinter import ttk
from tkinter import filedialog


def verify_gui_input(gui):
    """Validate all GUI inputs"""
    data_sources = set(gui.tuple_of_data_sources)
    channel_inputs = {gui.input_start.get(), gui.input_stop1.get(), gui.input_stop2.get()}

    if gui.input_start.get() != 'PMT1':
        if gui.input_stop1.get() != 'PMT1':
            if gui.input_stop2.get() != 'PMT1':
                raise BrokenPipeError('PMT1 value has to be entered in inputs.')

    if gui.num_of_frames.get() == '':
        if 'Frames' not in data_sources:
            raise BrokenPipeError('You must either enter a frame channel or number of frames.')
    else:
        if float(gui.num_of_frames.get()) != int(float(gui.num_of_frames.get())):
            raise ValueError('Please enter an integer number of frames.')

    if int(gui.num_of_frames.get()) < 0:
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

    list_of_outputs = re.sub(r'\s', '', gui.outputs.get()).split(',')
    if not set(list_of_outputs).issubset({'tiff', 'tif', 'array', 'single'}):
        raise UserWarning('Wrong input detected.')


class GUIApp(object):
    """Main GUI for the multiscaler code"""
    def __init__(self):
        self.root = Tk()
        self.root.title("Multiscaler Readout and Display")
        self.root.rowconfigure(5, weight=1)
        self.root.columnconfigure(5, weight=1)

        # Part containing the browse for file option
        main_frame = ttk.Frame(self.root, width=800, height=800)
        main_frame.grid(column=0, row=0)
        main_frame['borderwidth'] = 2

        self.filename = StringVar()

        browse_button = ttk.Button(main_frame, text="Browse", command=self.__browsefunc)
        browse_button.grid(column=0, row=0, sticky='ns')

        # Part containing the data about input channels
        # Conboboxes
        self.input_start = StringVar()
        self.input_stop1 = StringVar()
        self.input_stop2 = StringVar()
        self.tuple_of_data_sources = ('PMT1', 'Lines', 'Frames', 'Laser', 'TAG Lens', 'Empty')  # TODO: No PMT2 currently
        mb1 = ttk.Combobox(main_frame, textvariable=self.input_start)
        mb1.grid(column=3, row=1, sticky='we')
        mb1.set('Frames')
        mb1['values'] = self.tuple_of_data_sources
        mb2 = ttk.Combobox(main_frame, textvariable=self.input_stop1)
        mb2.grid(column=3, row=2, sticky='we')
        mb2.set('PMT1')
        mb2['values'] = self.tuple_of_data_sources
        mb3 = ttk.Combobox(main_frame, textvariable=self.input_stop2)
        mb3.grid(column=3, row=3, sticky='we')
        mb3.set('Lines')
        mb3['values'] = self.tuple_of_data_sources

        # Labels
        input_channel_1 = ttk.Label(main_frame, text='START')
        input_channel_1.grid(column=0, row=1, sticky='ns')
        input_channel_2 = ttk.Label(main_frame, text='STOP1')
        input_channel_2.grid(column=0, row=2, sticky='ns')
        input_channel_3 = ttk.Label(main_frame, text='STOP2')
        input_channel_3.grid(column=0, row=3, sticky='ns')

        # Number of frames in the data
        frame_label = ttk.Label(main_frame, text='Number of frames')
        frame_label.grid(column=0, row=4, sticky='ns')

        self.num_of_frames = StringVar(value=1)
        num_frames_entry = ttk.Entry(main_frame, textvariable=self.num_of_frames)
        num_frames_entry.grid(column=0, row=5, sticky='ns')

        # Wanted outputs
        outputs_label = ttk.Label(main_frame, text='Outputs ["tiff", "array", "single"]')
        outputs_label.grid(column=0, row=6, sticky='ns')

        self.outputs = StringVar(value='single')
        outputs_entry = ttk.Entry(main_frame, textvariable=self.outputs)
        outputs_entry.grid(column=0, row=7, sticky='ns')

        # Define image sizes
        image_size_label = ttk.Label(main_frame, text='Image sizes')
        image_size_label.grid(column=6, row=0, sticky='ns')
        x_size_label = ttk.Label(main_frame, text='X')
        x_size_label.grid(column=5, row=1, sticky='ns')
        y_size_label = ttk.Label(main_frame, text='Y')
        y_size_label.grid(column=6, row=1, sticky='ns')
        z_size_label = ttk.Label(main_frame, text='Z')
        z_size_label.grid(column=7, row=1, sticky='ns')

        self.x_pixels = StringVar(value=256)
        self.y_pixels = StringVar(value=512)
        self.z_pixels = StringVar(value=100)

        x_pixels_entry = ttk.Entry(main_frame, textvariable=self.x_pixels)
        x_pixels_entry.grid(column=5, row=2, sticky='ns')
        y_pixels_entry = ttk.Entry(main_frame, textvariable=self.y_pixels)
        y_pixels_entry.grid(column=6, row=2, sticky='ns')
        z_pixels_entry = ttk.Entry(main_frame, textvariable=self.z_pixels)
        z_pixels_entry.grid(column=7, row=2, sticky='ns')

        # Laser repetition rate
        laser1_label = ttk.Label(main_frame, text='Laser nominal rep. rate (FLIM)')
        laser1_label.grid(column=6, row=3, sticky='ns')
        laser2_label = ttk.Label(main_frame, text='Pulses per second')
        laser2_label.grid(column=7, row=4, sticky='ns')

        self.reprate = StringVar(value=0)  # 80e6 for the Chameleon, 0 to raise ZeroDivisionError
        reprate_entry = ttk.Entry(main_frame, textvariable=self.reprate)
        reprate_entry.grid(column=6, row=4, sticky='ns')

        # Binwidth of Multiscaler (for FLIM)
        binwidth_label = ttk.Label(main_frame, text='Binwidth of Multiscaler [sec]')
        binwidth_label.grid(column=6, row=5, sticky='ns')
        self.binwidth = StringVar(value=800e-12)
        binwidth_entry = ttk.Entry(main_frame, textvariable=self.binwidth)
        binwidth_entry.grid(column=6, row=6, sticky='ns')

        # TAG nominal frequency
        tag_label = ttk.Label(main_frame, text='TAG nominal frequency [Hz]')
        tag_label.grid(column=6, row=7, sticky='ns')
        self.tag_freq = StringVar(value=0.1897e6)
        tag_label_entry = ttk.Entry(main_frame, textvariable=self.tag_freq)
        tag_label_entry.grid(column=6, row=8, sticky='ns')

        # Define the last quit button and wrap up GUI
        quit_button = ttk.Button(self.root, text='Start', command=self.root.destroy)
        quit_button.grid()

        # self.root.bind('<Return>', quit_button)
        for child in main_frame.winfo_children():
            child.grid_configure(padx=2, pady=2)
        self.root.wait_window()

    def __browsefunc(self):
        self.filename.set(filedialog.askopenfilename(filetypes=[('List files', '*.lst')], title='Choose a list file'))

if __name__ == '__main__':
    app = GUIApp()

