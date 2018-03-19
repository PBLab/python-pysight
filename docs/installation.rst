============
Installation
============

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
