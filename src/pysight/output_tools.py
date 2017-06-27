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



    return output_list


def gen_tiff(movie, gui) -> int:
    """
    Algorithm's output will be a tiff file.
    """
    movie.create_tif()
    print('Tiff stack created with name {}.tif, \none for each channel.'.format(gui.filename.get()[:-4]))
    return 1


def gen_array(movie, gui) -> List:
    """
    Algorithm's output will be a list, each cell containing the full array of data of that channel
    """
    chan_data_list = movie.create_array()
    print('Data array created.')
    return chan_data_list


def gen_single(movie, gui) -> List[np.ndarray]:
    """
    Algorithm's output will be list, with a single stack for each channel.
    """
    chan_data_list: List[np.ndarray] = movie.create_single_volume('all')
    print(r'A summed array of all volumes\frames was created.')
    return chan_data_list

