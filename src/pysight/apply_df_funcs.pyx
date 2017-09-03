# encoding: utf-8
cimport cython

cimport numpy as np
import numpy as np

cdef dict hex_to_bin_dict():

    cdef dict diction = {\
            '0': '0000',
            '1': '0001',
            '2': '0010',
            '3': '0011',
            '4': '0100',
            '5': '0101',
            '6': '0110',
            '7': '0111',
            '8': '1000',
            '9': '1001',
            'a': '1010',
            'b': '1011',
            'c': '1100',
            'd': '1101',
            'e': '1110',
            'f': '1111',
        }

    return diction


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef list iter_string_hex_to_bin(str long_str):
    cdef unsigned long long idx
    cdef dict diction
    cdef list result

    result = []
    diction = hex_to_bin_dict()

    for idx in range(len(long_str)):
        result.append(diction[long_str[idx]])

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray convert_bin_to_int(np.ndarray arr):
    cdef np.ndarray[np.uint64_t, ndim=1] result = np.zeros((len(arr),), dtype=np.uint64)
    cdef unsigned long long idx
    cdef unsigned long long length = len(arr)

    for idx in range(length):
        result[idx] = int(arr[idx], 2)

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray convert_hex_to_int(np.ndarray arr):
    cdef np.ndarray[np.uint64_t, ndim=1] result = np.zeros((len(arr),), dtype=np.uint64)
    cdef unsigned long long idx
    cdef unsigned long long length = len(arr)

    for idx in range(length):
        result[idx] = int(arr[idx], 16)

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray convert_hex_to_bin(np.ndarray arr):
    cdef np.ndarray[object, ndim=1] result = np.empty((len(arr),), dtype=object)
    cdef unsigned long long idx
    cdef unsigned long long length = len(arr)

    for idx in range(length):
        result[idx] = bin(int(arr[idx], 16))[2:]

    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple get_lost_bit_np(list list_with_lost, int step_size, unsigned long long num_of_events):
    cdef list list_of_lost_bits
    cdef np.ndarray[np.uint64_t, ndim=1] timings = np.zeros((num_of_events,), dtype=np.uint64)
    cdef unsigned long long idx, lin_idx
    cdef str str_as_hex

    list_of_lost_bits = []

    for lin_idx, idx in enumerate(range(0, len(list_with_lost), step_size)):
        list_of_lost_bits.append(list_with_lost[idx][0])

        str_as_hex = "".join(list_with_lost[idx:idx+step_size])[1:]
        timings[lin_idx] = int(str_as_hex, 16)

    return list_of_lost_bits, timings


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tuple get_lost_bit_tag(list list_with_lost, int step_size, unsigned long long num_of_events):
    cdef list list_of_lost_bits
    cdef np.ndarray[object, ndim=1] timings = np.empty((num_of_events,), dtype=object)
    cdef unsigned long long idx, lin_idx
    cdef str str_as_hex

    list_of_lost_bits = []

    for lin_idx, idx in enumerate(range(0, len(list_with_lost), step_size)):
        list_of_lost_bits.append(list_with_lost[idx][0])
        timings[lin_idx] = "".join(list_with_lost[idx:idx+step_size])[1:]

    return list_of_lost_bits, timings
