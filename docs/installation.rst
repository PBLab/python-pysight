============
Installation
============

Due to some conflicting package dependencies and general Pythonian mess with packaging, installation works as follows:

Users without Python installed:
-------------------------------
Download and install Anaconda_ for Python 3.6.

.. _Anaconda: https://www.continuum.io/downloads

It's usually good habit to create a new environment for new projects. At the command line:
::
    conda create --name py36 python=3.6 numba -c conda-forge
    source activate py36
    pip install pysight


Users who already installed Python
----------------------------------
In a virtual environment install two of PySight's dependencies first, then install the package itself::
    pip install numba
    pip install pysight

The "Usage" tab provides more details on the operation of PySight.
