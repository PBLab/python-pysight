============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/PBLab/python-pysight/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

PySight could always use more documentation, whether as part of the
official PySight docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/PBLab/python-pysight/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

Development
===========

To set up PySight for local development:

1. Fork `python-pysight <https://github.com/PBLab/python-pysight>`_
   (look for the "Fork" button).
2. Clone your fork locally::

    git clone git@github.com:your_name_here/python-pysight.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

5. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests [1]_.
2. Format the code using `black <https://black.readthedocs.io/en/stable/index.html>`_: ``black python-pysight --py36``.
3. Update documentation when there's new API, functionality, etc.
4. Add a note to ``CHANGELOG.rst`` about the changes.
5. Add yourself to ``AUTHORS.rst``.

.. [1] If you don't have all the necessary Python versions available locally you can rely on Travis - it will
       `run the tests <https://travis-ci.org/PBLab/python-pysight/pull_requests>`_ for each change you add in the pull request.

       It will be slower though ...

Releasing a New version
-----------------------

To create and publish a new version, once all code changes are made, follow these steps:

1. Make sure that the ``dev`` installation of PySight is installed.
2. ``black .`` from the project's home directory.
3. ``bumpversion --allow-dirty patch`` to bump the version. Other options include ``minor`` and ``major``.
4. ``python setup.py clean --all sdist bdist_wheel``
5. ``twine upload --skip-existing dist/*``