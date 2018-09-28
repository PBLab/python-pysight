#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import os
import re
import pathlib
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()

# Enable code coverage for C code: we can't use CFLAGS=-coverage in toxa.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in toxa.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if 'TOXENV' in os.environ and 'SETUPPY_CFLAGS' in os.environ:
    os.environ['CFLAGS'] = os.environ['SETUPPY_CFLAGS']

setup(
    name='pysight',
    version='0.9.5',
    license='Free for non-commercial use',
    description='Create images and volumes from photon lists generated by a multiscaler',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Hagai Har-Gil',
    author_email='hagaihargil@protonmail.com',
    url=r'https://github.com/PBLab/python-pysight/',
    packages=find_packages('pysight'),
    py_modules=[splitext(basename(path))[0] for path in glob('pysight/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    keywords=[
        'multiscaler', 'photon counting'
    ],
    install_requires=[
        'numpy < 1.16',
        'matplotlib < 2.3',
        'pandas < 0.24',
        'attrs < 18.2',
        'cython < 0.29',
        'tables < 3.5',
        'scipy < 1.2',
        'scikit-learn < 0.20',
        'h5py < 2.9',
        'h5py-cache < 1.1',
        'tqdm < 4.24',
        'numba < 0.41',
        'ansimarkup < 1.5',
        'psutil < 5.5',
    ],
    setup_requires=[
        'numpy'
    ],
    data_files=['pysight' + os.sep + 'configs' + os.sep + 'default.json',
                str(pathlib.Path('./mcs6a_settings_files/pre_test_2d_a.set')),
                str(pathlib.Path('./mcs6a_settings_files/pre_test_3d_a.set'))]
)
