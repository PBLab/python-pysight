import sys, ctypes
from ctypes import c_uint32, Structure


class DataInChannels(Structure):
    _fields_ = [("x", c_uint32),
                ("y", c_uint32)]
