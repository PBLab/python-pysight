============
Installation
============

---------------------
Software Installation
---------------------

Users without Python installed:
-------------------------------
Download and install Anaconda_ for Python 3.6.

.. _Anaconda: https://www.continuum.io/downloads

It's usually good habit to create a new environment for new projects. At the command line:
::
    conda create --name py36 python=3.6 -c conda-forge
    source activate py36
    pip install pysight

Users who already installed Python
----------------------------------
In a virtual environment simply install PySight::

    pip install pysight

In some environments you may be required to install ``numpy`` before installing PySight.

The "Usage" page provides more details on the operation of PySight.

---------------------
Hardware Installation
---------------------

While PySight can parse any time-tagged list of photon arrival times, it will usually be paired with a
FAST COMTEC MCS6A Multiscaler. Thus, installing and running the multiscaler is a prerequisite before
running PySight. `This <https://www.fastcomtec.com/ftp/manuals/mcs6adoc.pdf>`_ is a link to the official multiscaler handbook,
with its quite simple installation instructions.

After you've managed to run a simple multiscaler experiment, you can use one of the settings files we supply with PySight,
located at the ``mcs6a_settings_files`` folder in the main repo, to see how a typical Multiscaler + PySight experiment looks like,
settings-wise.
