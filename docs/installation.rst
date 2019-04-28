============
Installation
============

---------------------
Software Installation
---------------------

Users without Python installed:
-------------------------------
Download and install Anaconda_ for Python 3.6+.

.. _Anaconda: https://www.continuum.io/downloads

It's usually good habit to create a new environment for new projects.
At the command line::

    conda create --name pysightenv python=3 -c conda-forge
    source activate pysightenv
    pip install pysight

Users who already installed Python
----------------------------------
In a virtual environment simply install PySight::

    pip install pysight

*Note for Windows users:* PySight uses Cython to compile C extensions. If this is the first time you're using Cython, you'll have
to download the `Visual Studio C++ Build Tools <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15>`_
and install them before you'll be able to ``pip install pysight``.

The "Usage" page provides more details on the operation of PySight.

---------------------
Hardware Installation
---------------------

While PySight can parse any time-tagged list of photon arrival times, it will usually be paired with a
FAST ComTec MCS6A Multiscaler. Thus, installing and running the multiscaler is a prerequisite before
running PySight. `This <https://www.fastcomtec.com/ftp/manuals/mcs6adoc.pdf>`_ is a link to the official multiscaler handbook,
with its quite simple installation instructions.

After you've managed to run a simple multiscaler experiment, you can use one of the settings files we supply with PySight,
located at the ``mcs6a_settings_files`` folder in the main repo, to see how a typical Multiscaler + PySight experiment looks like,
settings-wise.

The installation step will probably involve re-routing the scanning
elements synchronization signals into the multiscaler. You can
consult the ScanImage manual (if you're using ScanImage) for the
specific ports in question. For example, a standard resonant-galvo
setup will need to feed the its line signal, `taken from PFI6 at the
respective breakbox <http://scanimage.vidriotechnologies.com/pages/viewpage.action?pageId=26509475>`_,
into one of the multiscaler analog inputs (STOP1, for example).

If you're unsure of the exact wiring for your specific scope, don't
hesitate to contact the package authors.
