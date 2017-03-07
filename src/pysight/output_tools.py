"""
__author__ = Hagai Hargil
"""
from typing import Dict
import re


def generate_output_list(movie, gui):
    """
    Main function that generates outputs based on given parameters.
    :return List containing outputs.
    """

    list_of_wanted_outputs = re.sub(r'\s', '', gui.outputs.get()).split(',')

    dict_of_outputs = inst_outputs()
    output_list = []
    for idx ,output in enumerate(list_of_wanted_outputs, 1):
        output_list.append(dict_of_outputs[output](movie, gui))
        print('{} of {} wanted outputs created.'.format(idx, len(list_of_wanted_outputs)))

    return output_list


def gen_tiff(movie, gui):
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


def gen_single(movie, gui):
    """
    Algorithm's output will be a single stack.
    """
    single_volume = movie.create_single_volume('all')
    print(r'A summed array of all volumes\frames was created.')
    return single_volume


def inst_outputs() -> Dict:
    """
    Create the dict of possible outputs.
    :return:
    """
    dict_of_outputs = {
        'tiff': gen_tiff,
        'tif': gen_tiff,
        'array': gen_array,
        'single': gen_single
    }

    return dict_of_outputs
