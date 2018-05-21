Changelog
=========

0.9.0 (2018-05-21)
------------------

* Added binary list file support.

0.8.4 (2018-05-20)
------------------

* Pinned versions of required libraries.

* Fixed bidirectional TAG stack generation.


0.8.3 (2018-05-13)
------------------

* Fixed bug in ``VolumeGenerator`` when having a single frame slice.

0.8.2 (2018-05-12)
------------------

* Unified configuration file keyword to be ``cfg_file``.

* Dramatically increased performance to due faster I/O.

* Refactored the ``Movie`` class, changing the ``Volume`` class with a ``FrameChunk`` class.

* Added tests.

* Updated docs.

* Changed many internal file names.

0.8.1 (2018-03-19)
------------------

* Bug fix in ``setup.py``.

0.8.0 (2018-03-19)
-----------------

* Added ``recursive`` and ``n_proc`` keywords to ``mp_batch``, and changed return type to ``None``.

* Changed source tree structure for better clarity.

* Renamed ``run_batch`` to ``run_batch_lst``.

* More internal improvements.

* Z-axis bins range is equal, i.e. each bin spans the same axial distance in microns.

* Travis CI is back on.

* Added option to run PySight with a predetermined config file: ``main.run(cfg_file='/path/to/file.json')``.

* New integration tests.

0.7.3 (2018-01-01)
------------------

* Bug fixes to ``run_batch``.

* New function ``mp_batch(foldername, glob_str)`` for parallel processing of a folder of list files.

0.7.2 (2018-01-01)
------------------

* Minor bug fixes.

0.7.1 (2018-01-01)
------------------

* GUI is now startable with "S" key and \ or "Enter".

* More tests to new SignalValidator class.

* Bug fix for the validation process.

* Making progress on multiprocessing support.

0.7.0 (2018-01-01)
------------------

* Refactoring and additions to GUI, including new choices between imaging systems.

* Better UI and UX.

* Not all tests pass.

0.6.34 (2017-12-27)
-------------------

* Changed output of ``run_batch`` to a DataFrame.

* Refactored ``tabulation_tools``.

0.6.33 (2017-12-26)
-------------------

* Hotfix to ``attrs`` problem in ``setup.py``.

0.6.32 (2017-12-26)
-------------------

* Trial with Numba and setuptools.

* Type annotations.

* Documentation update.

* Fixes for single-photon bug.

0.6.31 (2017-12-26)
-------------------

* Bug fix for empty volumes with multichannel support.

0.6.30 (2017-12-25)
-------------------

* Fixed another bug with the line handling.

* Fixed a bug with a missing PMT channel.

0.6.29 (2017-12-25)
-------------------

* Code cleanups.

* More adjustments to line handling in bidirectional mode.

0.6.28 (2017-12-25)
-------------------

* Better handling of line signal.

0.6.27 (2017-12-24)
-------------------

* Separated handling of unidir and bidir corrupt line signals.

* Refactored line signal handling module.

* Added multiple tests to line signal handling.

0.6.26 (2017-12-21)
-------------------

* Missing line signals take mirror phase into account.

* Fixes for MScan system.

0.6.25 (2017-12-20)
-------------------

* Even more edge-case handling.

0.6.24 (2017-12-20)
-------------------

* Deals with more edge-cases in missing line signals.

0.6.23 (2017-12-20)
-------------------

* Fixed bugs with interpolations and TAG signals.

0.6.22 (2017-12-19)
-------------------

* Added interpolation for missing line signals.

0.6.21 (2017-12-19)
-------------------

* More work on TAG interpolation.

0.6.20 (2017-12-17)
-------------------

* Fixed a bug with TAG lens interpolation.

0.6.19 (2017-12-06)
-------------------

* Bug with lines allocation in the ``Volume`` object following an API change in pandas.

* Allows for single frame experiments.

0.6.18 (2017-12-05)
-------------------

* Fixed a bug with bidirectional scanning.

* Possible fix for data that don't have lines since the beginning of the experiment.

0.6.17 (2017-12-04)
-------------------

* Fixed a bug with the filename of the ``DEBUG``ged version.

0.6.16 (2017-11-20)
-------------------

* Support for non-phase allocation of TAG pulses.

* Removal of old TAG module.

* ``run_batch()`` works without choosing a mock list file.

0.6.15 (2017-11-05)
-------------------

* Better bidirectional support.

0.6.14 (2017-10-30)
-------------------

* Added a ``glob_str`` and ``recursive`` parameters to ``run_batch()``.

* Added a ``DEBUG`` suffix to files generated when debugging.

* Changed license to creative commons.

* Small bug fixes, somewhat decreased memory usage.

* Improved bidirectional scanning performance and robustness by reworking its mechanism.

0.6.13 (2017-10-08)
-------------------

* The TAG phase is now between 0 and 1, generating non-cyclic volumes.

0.6.12 (2017-10-08)
-------------------

* Removed the experimental ``parallel`` feature from the Numba implementation.

* Fixed bidirectional image generation.

* Default fill fraction is now 75% to better suit ScanImage's defaults.

0.6.11 (2017-10-06)
-------------------

* Complete re-write of TAG lens processing module.

0.6.10 (2017-10-03)
-------------------

* Fixed a bug occurring when TAG lens interpolation fails.

* Discovered another bug with the interpolation process which is currently unresolved.

* Fixed small issue with a TAG test function.

0.6.9 (2017-09-29)
------------------

* Stacking the final array is now an order-of-magnitude faster - the first dimension is now considered `time`.

* Fixed a bug with singleton dimensions.

* Fixed a bug with no "In Memory" output.

0.6.8 (2017-09-28)
------------------

* Small bug fix in progress bar.

0.6.7 (2017-09-28)
------------------

* Faster I/O.

* Datasets are now ``uint8`` (full stack) and ``uint16`` (summed stack).

* Allowing outputs without the "In Memory" requirement.

* Added a progress bar.

* ``show_summed()`` works, ``show_stack()`` might not.

0.6.6 (2017-09-27)
------------------

* Now compressing HDF5 files.

* Fixed small bug in TAG implementation.

0.6.5 (2017-09-18)
------------------

* Writing output ``.hdf5`` to disk is much faster now.

0.6.4 (2017-09-18)
------------------

* Fixed bug with two-channel output.

* Fixed bug with "early" photons.

0.6.3 (2017-09-11)
------------------

* Better support for "early" photons.

* Allow for no outputs from PySight.

0.6.2 (2017-08-29)
------------------

* Fixed bug with ``movie.show_stack()``.

0.6.1 (2017-08-28)
------------------

* Added gating to photons that arrive too early (or too late) after a laser pulse.

0.6.0 (2017-08-27)
------------------

* Changed output file format to ``.hdf5`` due to compatibility issues of ``.tif``s.

0.5.25 (2017-08-26)
-------------------

* Added the ``photons_per_pulse`` property to ``Movie()``.

* Introduced the ``run_batch(foldername)`` function to the ``main`` module, to run PySight with the same configs on multiple ``.lst`` files in a folder.

* Added the ``num_of_vols`` property to ``Movie()``.

0.5.24 (2017-07-30)
-------------------

* Bugfixes for line validations.

* Added methods ``show_summed(channel)`` and ``show_stack(channel, iterable)``.

* More refactoring to decrease class absolute size.

* Small bug fix in the sweeps-as-lines implementation.

0.5.23 (2017-07-20)
-------------------

* Supports generating images from pure sweeps, without a line signal.

* Supports generating images from combined sweep and line signals.

0.5.22 (2017-07-17)
-------------------

* Added an optional line frequency entry to the GUI.

* Refactoring of some parts of the validation tools.

* Small performance upgrade.

* Added an option to treat sweeps as lines.

0.5.21 (2017-07-07)
-------------------

* Added the acquisition delay and "hold-after" times to the calculation of the the absolute time of each event.

* Decreased package size dramatically by deleting unneeded test data.

* All 34 tests of code pass. I'll try to keep it that way :)

* Added an extrapolation method to create fake lines when the line data is too corrupt to work with. This is done using
  the new "line frequency" and "frame delay" parmaeters in the GUI.

0.5.20 (2017-07-01)
-------------------

* Refactored the output-generating script, while changing the possible outputs of PySight:
    * Summed tif.
    * Full stack as tif.
    * In memory - both stack and tif accessed through the ``movie`` object.

0.5.19 (2017-06-29)
-------------------

* Fixed small bug with censoring.

* Added checks to see whether we need censor correction.

* Added tests for ``lst_tools`` - they should pass, much like ``file_io``'s tests.

0.5.18 (2017-06-29)
-------------------

* Added metadata from ``.lst`` file to the saved ``.tif`` file. Variables saved:
    * "fstchan"
    * "holdafter"
    * "periods"
    * "rtpreset"
    * "cycles"
    * "sequences"
    * "range"
    * "sweepmode"
    * "fdac"

0.5.17 (2017-06-29)
-------------------

* Fixed ``.tif`` generation.

* Refactoring of ``FileIO`` (tests still pass).

0.5.16 (2017-06-27)
-------------------

* Fixed small bug with censor correction.

0.5.15 (2017-06-27)
-------------------

* Refactoring of output:

    * Start of censor correction is integrated into the generation of the outputs.

    * More efficient when required to output several types of data.

0.5.14 (2017-06-26)
-------------------

* Bug fixes and performance improvements.

0.5.13 (2017-06-26)
-------------------

* Added SciPy dependency.

* Added nanoFLIM histogramming.

0.5.12 (2017-06-22)
-------------------

* Fixed small bug with GUI.

* Possible fix to TAG lens interpolation.

0.5.11 (2017-06-22)
-------------------

* Added FLIM functionality with laser clock in the Multiscaler's clock.

0.5.10 (2017-06-12)
-------------------

* Changes and optimizations for the file IO process.

* Fixed a bug with laser pulses verification.

* Added offset parameter for laser input.

0.5.9 (2017-06-11)
------------------

* Much (MUCH) faster concatenation of the data.

* Fixed a bug with the number of empty histograms that were added to the learning dataset.

0.5.8 (2017-06-08)
------------------

* Robustness upgrades.

* QOL changes to GUI.

* A "power" number is needed for ``learn_histograms()`` - the percent of power given to the Qubig. It's just for saving, labeling is done with the ``label`` keyword.

* A ``foldername`` to which the data will be saved to has also been added.

0.5.7 (2017-06-08)
------------------

* More robust data generation.

* Added ``scikit-learn`` to ``requirements.txt`` and ``setup.py``.

* ``CensorCorrection().learn_histograms()`` now receives the power label as its input - must be an integer.

* Return of ``CensorCorrection().learn_histograms()`` is now ``data, labels``.

* Made ``__get_bincount_deque()`` private. To be accessed using ``learn_histograms()`` only.

0.5.6 (2017-06-08)
------------------

* ML classification is functional.

0.5.5 (2017-06-07)
------------------

* Bug fixes for single-pixel frames

* Bug fixes for defining amount of frames manually in script.

* Censor correction shouldn't require as much memory as it did. It's still not as fast as it can be.

* Loading a configuration file will make it the "last used" file, reloading it when re-running PySight.

0.5.4 (2017-06-06)
------------------

* Fixed untested typo.

0.5.3 (2017-06-06)
------------------

* Bug fixes, including support for single-pixel images.

* Script should require less memory while running.

0.5.2 (2017-06-06)
------------------

* Added basic support for "Censor Correction".

0.5.1 (2017-06-04)
------------------

* Another go at Linux namespace conflicts.

0.5.0 (2017-06-04)
------------------

* Added the ``CensorCorrection`` class for processing generated data using the censor correction method. Current available methods are:
    * ``censored.gen_bincount_deque()``: Bin the photons into their relative laser pulses, and count how many photons arrived due to each pulse.
    * ``censored.find_temp_structure_deque()``: Generate a summed histogram of the temporal structure of detected photons.

* Fixed linux bug with ``Deque`` import.

* Added tests.

0.4.8 (2017-05-31)
------------------

* Added type hinting. As a result, disabled support for Python version 3.5. Code is now entirely 3.6-dependent.

* Added ``.json`` configuration files to the GUI. It also automatically loads the last modified configuration file.

* Updated docs.

0.4.7 (2017-05-25)
------------------

* Fixed some of the tests.

* Added option to save or discard photons arriving during the returning phase of a unidirectional scan. This is the default option now.

* Introduced ``Fill Fraction`` parameter that determines the amount of `time` the mirrors spend "inside" the image.

* Some tests are working again.

* Many other bugfixes.

0.4.6 (2017-05-16)
------------------

* Use ``Debug?`` to read a small portion of an ``.lst`` file.

* Changed defaults in GUI.

* Allows acquisition in bi-directional scanning mode. This is enabled with the ``Mirror phase`` and ``Flyback`` parameters in the GUI.

* Backend changes for possible future support of binary files.

* The code allows to dismiss unwanted input channels by specifying them as "Empty".
    * If you mark a channel as containing data while it's inactive, an error will terminate execution.

* Massive refactoring of pipeline.

0.4.5 (2017-04-17)
------------------

* Bug fixes and improvements to TAG lens interpolation.

0.4.4 (2017-04-08)
------------------

* Changes to file I\O.

* Number of requested frames should actually matter now.

* GUI improvements.

0.4.3 (2017-04-02)
------------------

* Removed Dask.

* Refactored class structure, remove the ``Frame`` class.

* Refactored GUI code.

0.4.2 (2017-03-30)
------------------

* Added Dask ``delayed`` interface.

0.4.1 (2017-03-30)
------------------

* Updates to setup.py to allow docs to build successfully.

* Small updates to docs.

* GUI improvements.

0.4.0 (2017-03-16)
------------------

* Changes file IO completely. Performance should be higher.

* TAG lens bug fixes.

* Updated docs.

* Updated tests.

0.3.6 (2017-03-14)
------------------

* Basic support for TAG bits - no actual interpolation yet.

* GUI additions and changes.

* Minor performance upgrades.


0.3.5 (2017-03-11)
------------------

* Added sinusoidal interpolation to TAG phase.

* Sorting is now only done for TAG lens input.

* Added ``fileIO_tools.py`` module for increased simplicity.

* Added more verifications to user inputs from GUI that pop up sooner, before heavy computation is made.

* Increased file IO speed with a new ``np.fromfile`` method.


0.3.4 (2017-03-09)
------------------

* More fixes to the [-1] vector problem.

* Added a ``sort`` function before handling the data, because of irregularities.


0.3.3 (2017-03-08)
------------------

* Code can take care of the the infamous [-1, ..., -1] index list.

* Added ``debug`` mode in which the algorithm reads only a limited amount of lines from a file.

* Fixed minor bug in ``__create_hist``.

* Decreased size of package by removing excess lines of data for tests.

0.3.2 (2017-03-07)
------------------

* Added verifications on the FLIM input.

* Bug fixes in FLIM implementation.

0.3.1 (2017-03-07)
------------------

* Tiffs are now saved untiled. Depth axis is x-axis.

* Installation should run smoothly if following the instructions.

0.3.0 (2017-03-07)
------------------

* Added method ``create_array`` to Movie() that returns a deque containing the raw data generated by the ``np.histogram`` function, for visualization and analysis purposes.

* Added method ``create_single_volume`` to Movie() that sums all stacks into a single array.

* Fixed bugs in ``tag_tools``, mainly in ``verify_periodicity()``.

* Allows for more elaborate user inputs, requiring to choose which type of output you wish for.

* Basic FLIM support.

0.2.0 (2017-03-05)
------------------

* Support for TAG lens added - phase interpolation and image display. Note: The algorithm currently assumes that the pulse is triggered at the zero-phase of the TAG lens.

* ``pip`` installation fixed by requiring Numba as a prerequisite.

* Number of pixels in the "Frame" direction (x) supersedes the number of frames as listed by the user.

* Due to massive changes, one test is currently broken.

0.1.7 (2017-03-01)
------------------

* Potential fix to ``pip install`` issues.

* Start of TAG lens interpolation support.

0.1.6 (2017-02-28)
------------------

* More tests coverage.

* Enforced a few types checks.

0.1.5 (2017-02-28)
------------------

* Single-lined frames are now supported.

0.1.4 (2017-02-28)
------------------

* Frames are now generated with a generator.
* Fix to installation problems of previous version.

0.1.3 (2017-02-28)
------------------

* Changed IO from ``.read()`` to ``.readlines()`` for better Linux compatibility.

* ``.tif`` is now saved frame-by-frame to save memory, and the method was renamed to ``create_tif()``.

0.1.2 (2017-02-27)
------------------

* Includes ``tifffile`` and minor improvements.

0.1.1 (2017-02-27)
------------------

* Bug fixes during installation of Numba.

* Added the ``run()`` method for ``main_multiscaler_readout``.

0.1.0 (2017-02-27)
------------------

* First release on PyPI.

