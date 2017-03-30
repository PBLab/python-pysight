============
Installation
============

Due to some conflicting package dependencies and general Pythonian mess with packaging, installation works as follows:

Download and install Anaconda_ for Python 3.6.

.. _Anaconda: https://www.continuum.io/downloads

At the command line:
::

    conda install numba
    conda install tifffile -c conda-forge
    pip install pysight


```tifffile``` is only necessary if you wish to save the data to a `.tif` file.

The "Usage" tab provides more details about actual usage of PySight.
