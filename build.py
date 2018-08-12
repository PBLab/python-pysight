import os
import sys


from distutils.core import Extension

from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)
from distutils.command.build_ext import build_ext


# C Extensions
extensions = [
    Extension('pysight.ascii_list_file_parser.apply_df_funcs',
                ['./pysight/ascii_list_file_parser/apply_df_funcs.c']),
    ]


class BuildFailed(Exception):

    pass


class ExtBuilder(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            print(
                "************************************************************"
            )
            print(
                "Cannot compile C accelerator module, use pure python version"
            )
            print(
                "************************************************************"
            )

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError,
                DistutilsPlatformError, ValueError):
            print(
                "************************************************************"
            )
            print(
                "Cannot compile C accelerator module, use pure python version"
            )
            print(
                "************************************************************"
            )


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update({
        'ext_modules': extensions,
        'cmdclass': {
            'build_ext': ExtBuilder
        }
})