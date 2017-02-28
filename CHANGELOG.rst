
Changelog
=========

0.1.0 (2017-02-27)
-----------------------------------------

* First release on PyPI.

0.1.1 (2017-02-27)
-----------------------------------------

* Bug fixes during installation of Numba.
* Added the ``run()`` method for ``main_multiscaler_readout``.

0.1.2 (2017-02-27)
-----------------------------------------

* Includes ``tifffile`` and minor improvements.

0.1.3 (2017-02-28)
-----------------------------------------

* Changed IO from ``.read()`` to ``.readlines()`` for better Linux compatibility.
* ``.tif`` is now saved frame-by-frame to save memory, and the method was renamed to ``create_tif()``.

0.1.4 (2017-02-28)
-----------------------------------------

* Frames are now generated with a generator.
* Fix to installation problems of previous version.


