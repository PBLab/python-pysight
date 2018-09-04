from pysight._native import ffi, lib


def test():
    return lib.read_lst(b'4-byte006.lst', 1554, 512, b'5')
