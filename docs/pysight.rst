Subpackages
-----------

.. toctree::

    pysight.ascii_list_file_parser
    pysight.binary_list_file_parser
    pysight.nd_hist_generator
    pysight.gui

Submodules
----------

pysight.main module
-------------------

.. automodule:: pysight.main
    :members:
    :undoc-members:
    :show-inheritance:
    :functions:

.. function:: main_data_readout(config)

    Main function that reads the lst file and processes its data.
    Should not be run independently - only from other "run_X" functions.


.. function:: run(cfg_file: str = None) -> PySightOuput
    Run PySight.

    :param str cfg_file: Optionally supply an existing configuration filename. Otherwise a GUI will open.

    :return PySightOutput: A special object which allows for easy access to the raw and processed data.


.. function:: run_batch_lst(foldername: str, glob_str: str = "*.lst", recursive: bool = False, cfg_file: str = "") -> pd.DataFrame
    Run PySight on all list files in the folder

    :param str foldername: - Main folder to run the analysis on.
    :param str glob_str: String for the `glob` function to filter list files
    :param bool recursive: Whether the search should be recursive.
    :param str cfg_file: Name of config file to use
    :return pd.DataFrame: Record of analyzed data


pysight.read\_lst module
------------------------

.. automodule:: pysight.read_lst
    :members:
    :undoc-members:
    :show-inheritance:

pysight.tkinter\_gui\_multiscaler module
----------------------------------------

.. automodule:: pysight.tkinter_gui_multiscaler
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: pysight
    :members:
    :undoc-members:
    :show-inheritance:
