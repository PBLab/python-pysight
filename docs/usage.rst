=====
Usage
=====

To use PySight do one of the following:

1. write a Python script containing::

    from pysight import main_multiscaler_readout


    df, movie = main_multiscaler_readout.run()

2. From the command line::

    python /path/to/pysight/dir/main_multiscaler_readout.py

3. For batch processing of multiple list files::

    from pysight import main_multiscaler_readout

    foldername: str = r'/path/to/folder/with/list/files'
    globstr: str = ''  # a glob-pattern string to filter files to parse. Default is '*.lst'
    recursive: bool = False  # Boolean whether to iterate over nested folders in the main folder

    main_multiscaler_readout.run_batch(foldername=foldername, globstr=globstr, recursive=recursive)

These command will open a GUI in which you'll have to choose a ``.lst`` file to parse.

If you use option #1, The algorithm will create the pandas DataFrame ``df`` containing all data, and a ``movie`` object with allocated data.
Outputs come in the form of in-memory numpy arrays and ``.hdf5`` compressed multidimensional files. The output options are:

* **In Memory** - The returned objects contain all photons and generated histograms. Use it if you wish to further process the data in Python. Access the data with the ``.hist`` field of the movie object.
* **Full Stack** - PySight will save to disk a multidimensional histogram, one per spectral channel, with all of the events recorded. Dimensions are [t, x, y, z, tau].
* **Summed Stack** - PySight will sum the t dimension of the Full Stack to received a summed projection over time of the entire experiment.

Obviously, if you run the script from the command line (option #2) or in batch mode you can only interact with the final ``.hdf5`` files.

GUI Options
-----------
.. image:: gui.png
   :scale: 75%

Running **PySight** will open GUI seen above.

Choosing a ``.lst`` file for analysis is done with the *Browse* button, located at the top-left corner of the GUI.

The *Input Channels* option allows you to specify what was the input device to all three (supported) of the analog inputs of the multiscaler.
Note that at least one of these fields must be a ``PMT1`` entry.

To the right, *Image Size* determines the shape of the output matrix from the algorithm. The X dimension corresponds to the line signal, and it should fit the original signal fed into the multiscaler.
Below it you can check whether the mirrors acquired data in a bi-directional fashion.
You should also specify, in case data was acquired uni-directionally, whether the algorithm should keep the photons arriving during that returning phase.
Below, specify the phase delay of the scanners and fill fraction, to cancel pixel-shift and remove the non-linear area of the image, located in the edges. Phase delay is only used in bidirectional mode.

At the bottom part of the GUI you can load a specific ``.json`` file to be used as a config file. A default ``default.json`` file is shipped with the package,
but other files can be saved and loaded easily, to allow easier use of the GUI. The full filename is also saved.

Some other options include:

* *Debug?*: Reads a relatively small portion of a file, allows for quick code-checking.

* *TAG Bit allocation*: When using the TAG bits you can determine what each bit is responsible for.

* *Laser repetition rate*: For FLIM.

* *Binwidth*: The binwidth, in seconds, of your model of the MCS6A.


Limitations
-----------

* List (``.lst``) files have to be saved in ``ASCII`` format, and not binary.

* The code currently supports only three input channels.
