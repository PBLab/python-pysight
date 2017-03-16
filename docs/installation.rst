============
Installation
============

Due to some conflicting package dependencies and general Pythonian mess with packaging, installation works as follows:


At the command line:
::

    conda install numba
    conda install tifffile -c conda-forge
    pip install pysight


```tifffile``` is only necessary if you wish to save the data to a `.tif` file.
