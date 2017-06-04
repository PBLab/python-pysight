"""
__author__ = Hagai Hargil
"""
from typing import List
import numpy as np



def generate_output_list(movie, gui) -> List:
    """
    Main function that generates outputs based on given parameters.
    :return List containing outputs.
    """

    output_list: List = []

    if 1 == gui.summed.get():
        output_list.append(gen_single(movie=movie, gui=gui))

    if 1 == gui.full.get():
        output_list.append(gen_array(movie=movie, gui=gui))

    if 1 == gui.tif.get():
        output_list.append(gen_tiff(movie=movie, gui=gui))

    return output_list


def gen_tiff(movie, gui) -> int:
    """
    Algorithm's output will be a tiff file.
    """
    movie.create_tif()
    print('Tiff stack created with name {}.tif'.format(gui.filename.get()[:-4]))
    return 1


def gen_array(movie, gui):
    """
    Algorithm's output will be the full array of data
    """
    data_array = movie.create_array()
    print('Data array created.')
    return data_array


def gen_single(movie, gui) -> np.ndarray:
    """
    Algorithm's output will be a single stack.
    """
    single_volume: np.ndarray = movie.create_single_volume('all')
    print(r'A summed array of all volumes\frames was created.')
    return single_volume

