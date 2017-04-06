=====
Usage
=====

To use PySight, write a script containing::

    from pysight import main_multiscaler_readout


    df, movie, outputs = main_multiscaler_readout.run()


This command will open a GUI in which you'll have to choose a ``.lst`` file to parse.
The algorithm will create the pandas DataFrame ``df`` containing all data, a ``movie`` object with allocated data, and a list ``outputs``.
In the GUI you can also choose the wanted ``output(s)`` in that list using the checkboxes:

* ``single`` - The algorithm will create a single ``n``-dimensional matrix that is the sum of all photon events in the list file. Each dimension corresponds to a recorded physical axis - frames, lines, laser pulse, etc.
* ``array`` - Full-blown array containing all available data in separate frames\volumes, no aggregation like in ``single``. Access it from the ``.hist`` field of the generated object.
* ``tiff`` - Outputs a ``.tif`` file with the same name as the file, in the same folder.

GUI Options
-----------

* ``Debug?``: Reads a relatively small portion of a file, allows for quick code-checking.

Limitations
-----------

* List (``.lst``) files have to be saved in ``ASCII`` format, and not binary.

* The code currently supports only three input channels, and a single PMT channel.
