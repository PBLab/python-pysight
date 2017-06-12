"""
__author__ = Hagai Hargil
"""
import attr
import numpy as np
import pandas as pd
from attr.validators import instance_of
from pysight.movie_tools import Volume, Movie
from collections import deque, namedtuple
from typing import Tuple, Union
from numba import jit, uint64, uint8, int64


@attr.s(slots=True)
class CensorCorrection(object):
    movie            = attr.ib(validator=instance_of(Movie))
    reprate          = attr.ib(validator=instance_of(float))
    binwidth         = attr.ib(validator=instance_of(float))
    laser_offset     = attr.ib(validator=instance_of(float))
    all_laser_pulses = attr.ib()

    @property
    def bins_bet_pulses(self) -> int:
        return int(np.ceil(1 / (self.reprate * self.binwidth)))

    @property
    def offset(self):
        return int(np.floor(self.laser_offset * 10**-9 / self.binwidth))

    def __gen_laser_pulses_deque(self) -> np.ndarray:
        """
        If data has laser pulses - return them. Else - simulate them with an offset
        """
        start_time = 0
        step = self.bins_bet_pulses
        volumes_in_movie = self.movie.gen_of_volumes()

        if self.all_laser_pulses == 0:  # no 'Laser' data was recorded
            for vol in volumes_in_movie:
                yield np.arange(start=start_time+self.offset, stop=vol.end_time,
                                step=step, dtype=np.uint64)

        else:
            for vol in volumes_in_movie:
                yield self.all_laser_pulses[(self.all_laser_pulses >= vol.abs_start_time-step) &
                                            (self.all_laser_pulses <= vol.end_time+step)] + self.offset

    def __get_bincount_deque(self):
        print("Movie object created. Generating the bincount deque...")
        bincount_deque = deque()
        laser_pulses_deque = self.__gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque))
            dig, bincount = censored.gen_bincount()
            pos_idx = np.where(dig >= 0)[0]
            dig = dig[pos_idx]
            pos_photons = censored.df.iloc[pos_idx, -1].values.T
            if len(pos_photons) == 0:
                data_dict = {'photon_hist'    : np.zeros((self.bins_bet_pulses, 1), dtype=np.uint8),
                             'bincount'       : bincount,
                             'num_empty_hists': bincount[0]}
                return data_dict

            photon_hist = np.zeros((self.bins_bet_pulses, pos_photons.shape[0]), dtype=np.uint8)
            for laser_idx, photon in enumerate(np.nditer(pos_photons)):
                start_time = censored.laser_pulses[dig[laser_idx]]
                try:
                    end_time = censored.laser_pulses[dig[laser_idx] + 1]
                except IndexError:  # photons out of laser pulses
                    continue
                else:
                    photon_hist[:, laser_idx] = np.histogram(photon, bins=np.arange(start_time, end_time + 1,
                                                                                    dtype='uint64'))[0]

            data_dict = {'photon_hist'    : photon_hist,
                         'bincount'       : bincount,
                         'num_empty_hists': bincount[0]}
            assert data_dict['num_empty_hists'] >= 0, 'Sum of bincount: {}, number of photons: {}'\
                .format(sum(bincount), laser_idx)
            bincount_deque.append(data_dict)

        return bincount_deque

    def find_temporal_structure_deque(self):
        temp_struct_deque = deque()
        laser_pulses_deque = self.__gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque),
                                      binwidth=self.binwidth, reprate=self.reprate)
            temp_struct_deque.append(censored.find_temp_structure())
        return temp_struct_deque

    def create_array_of_hists_deque(self):
        """
        Go through each volume in the deque and find the laser pulses for each pixel, creating a summed histogram per pixel.
        :return:
        """
        temp_struct_deque = deque()
        laser_pulses_deque = self.__gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque),
                                      binwidth=self.binwidth, reprate=self.reprate)
            temp_struct_deque.append(censored.gen_array_of_hists())
        return temp_struct_deque

    def __gen_labels(self, size: int, label: Union[int, float]) -> np.ndarray:
        """
        Create labels for the ML algorithm. Label value must be an integer.
        :size: Number of elements
        :return: np.ndarray
        """
        if isinstance(label, int):  # fixed power during the session
            return np.ones(size, dtype=np.uint8) * label
        elif isinstance(label, float):  # `label` contains the frequency of the triangular wave
            pass

    def learn_histograms(self, label: Union[int, float], power: int, folder_to_save: str):
        """
        Implement the machine learning algorithm on the data.
        :param label: Label of ML algorithm.
        :param power: How much power was injected to the Qubig. For saving the file.
        :return: data, labels
        """
        from sklearn import svm, metrics
        import pathlib

        # Start by generating the data and arranging it properly for the machine
        bincount = self.__get_bincount_deque()
        print("Bincount done. Adding all data to a single matrix.")

        data = np.empty((self.bins_bet_pulses, 0))
        for vol in bincount:
            data = np.concatenate((data, vol['photon_hist']), axis=1)  # the histograms with photons in them
            data = np.concatenate((data, np.zeros((self.bins_bet_pulses, vol['num_empty_hists']),
                                                  dtype=np.uint8)), axis=1)  # empty hists
        data = data.T
        n_samples = data.shape[0]
        labels = self.__gen_labels(n_samples, label)
        classifier = svm.SVC(gamma=0.001)
        labels[1] = 10  # toying around

        print("Fitting the data...")
        classifier.fit(data[:n_samples // 2], labels[:n_samples // 2])

        # Predictions
        expected = labels[n_samples // 2:]
        predicted = classifier.predict(data[n_samples // 2:])
        print("Number of samples is %s." % n_samples)
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

        # Save the data for future use
        folder_as_path = pathlib.Path(folder_to_save)
        filename = str(folder_as_path / "{}p_label_{}.npy".format(power, label))
        self.__save_data(data=data, filename=filename)
        return data, labels

    def __save_data(self, data: np.ndarray, filename: str):
        """
        Save the data array for future training.
        :param data: Data to be learnt.
        :param filename: Including dir
        :return:
        """
        print("Saving to {}...".format(filename))
        with open(filename, 'wb') as f:
            np.save(f, data)


@attr.s(slots=True)
class CensoredVolume(object):
    df = attr.ib(validator=instance_of(pd.DataFrame))
    vol = attr.ib(validator=instance_of(Volume))
    laser_pulses = attr.ib(validator=instance_of(np.ndarray))
    offset = attr.ib(validator=instance_of(int))
    binwidth = attr.ib(default=800e-12)
    reprate = attr.ib(default=80e6)

    def gen_bincount(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin the photons into their relative laser pulses, and count how many photons arrived due to each pulse.
        """
        hist, _ = np.histogram(self.vol.data.loc[:, 'time_rel_frames'].values, bins=self.laser_pulses)
        dig = np.digitize(self.vol.data.loc[:, 'time_rel_frames'].values,
                          bins=self.laser_pulses) - 1
        return dig, np.bincount(hist)

    def find_temp_structure(self) -> np.ndarray:
        """
        Generate a summed histogram of the temporal structure of detected photons.
        """
        bins = np.arange(start=0, stop=np.ceil(1/(self.reprate * self.binwidth)) + 1, step=1)
        subtracted_times, _, _ = self.sort_photons_in_pulses()
        hist, _ = np.histogram(subtracted_times, bins=bins)
        return hist

    def sort_photons_in_pulses(self):
        """
        Helper function to generate a searchsorted output of photons in laser pulses.
        """
        pulses = self.laser_pulses
        sorted_indices: np.ndarray = np.searchsorted(pulses, self.vol.data['time_rel_frames'].values) - 1
        array_of_laser_starts = pulses[sorted_indices]
        subtracted_times = self.vol.data['time_rel_frames'].values - array_of_laser_starts
        return subtracted_times, array_of_laser_starts, sorted_indices

    def gen_array_of_hists(self) -> np.ndarray:
        """
        For a specific frame, sort photons and laser pulses inside the pixels to gain
        statistics on the distribution of the photons inside the pixels.
        :return: np.ndarray of the same size as the original image. Each pixels contains
        a histogram inside it.
        """
        BinData = namedtuple('BinData', ('hist', 'pulses', 'photons'))
        all_pulses = 0
        all_photons = 0

        hist, edges = self.vol.create_hist()
        # Create a relative timestamp to the line signal for each laser pulse
        sorted_pulses = np.searchsorted(edges[0][:-1], self.laser_pulses) - 1
        pulses = pd.DataFrame(data=self.laser_pulses[np.where(sorted_pulses >= 0)[0]], columns=['time_rel_frames'])
        pulses = pulses.assign(Lines=edges[0][:-1][sorted_pulses[np.where(sorted_pulses >= 0)[0]]])
        pulses.dropna(how='any', inplace=True)
        pulses.loc[:, 'Lines'] = pulses.loc[:, 'Lines'].astype('uint64')
        pulses.loc[:, 'time_rel_line'] = pulses.loc[:, 'time_rel_frames'] - pulses.loc[:, 'Lines']
        pulses.loc[:, 'Lines'] = pulses.loc[:, 'Lines'].astype('category')
        pulses.set_index(keys=['Lines'], inplace=True, append=True, drop=True)

        # Allocate laser pulses and photons to their bins
        pulses.loc[:, 'bins_x'] = (np.digitize(pulses.loc[:, 'time_rel_frames'].values,
                                   bins=edges[0]) - 1).astype('uint16', copy=False)
        pulses.loc[:, 'bins_y'] = (np.digitize(pulses.loc[:, 'time_rel_line'].values,
                                   bins=edges[1]) - 1).astype('uint16', copy=False)
        self.vol.data.loc[:, 'bins_x'] = (np.digitize(self.vol.data.loc[:, 'time_rel_frames'].values,
                                          bins=edges[0]) - 1).astype('uint16', copy=False)
        self.vol.data.loc[:, 'bins_y'] = (np.digitize(self.vol.data.loc[:, 'time_rel_line'].values,
                                          bins=edges[1]) - 1).astype('uint16', copy=False)
        pulses.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)
        self.vol.data.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)

        # Go through each bin and histogram the photons there
        image_bincount = np.zeros_like(hist, dtype=object)
        for row in range(self.vol.x_pixels):
            row_pulses = pulses.xs(key=row, level='bins_x', drop_level=False)
            assert len(row_pulses) > 0, 'Row {} contains no pulses'.format(row)
            try:
                row_photons = self.vol.data.xs(key=row, level='bins_x', drop_level=False)
            except KeyError:  # no photons in row
                for col in range(self.vol.y_pixels):
                    final_pulses = row_pulses.xs(key=col, level='bins_y', drop_level=False)
                    hist = (np.histogram(np.array([]), bins=final_pulses.loc[:, 'time_rel_line'].values)[0])\
                        .astype('uint8')
                    cur_bincount = np.bincount(hist).astype('uint64', copy=False)
                    tot_pulses = np.sum(cur_bincount)
                    tot_photons = np.average(cur_bincount, weights=range(len(cur_bincount)))
                    image_bincount[row, col] = BinData(hist=hist, pulses=tot_pulses, photons=tot_photons)
                    all_photons += tot_photons
                    all_pulses += tot_pulses
            else:
                for col in range(self.vol.y_pixels):
                    final_pulses = row_pulses.xs(key=col, level='bins_y', drop_level=False)
                    assert len(final_pulses) > 0, 'Column {} in row {} contains no pulses'.format(col, row)
                    try:
                        final_photons = row_photons.xs(key=col, level='bins_y', drop_level=False)
                    except KeyError:  # no photons in col
                        hist = (np.histogram(np.array([]), bins=final_pulses.loc[:, 'time_rel_line'].values)[0])\
                            .astype('uint8', copy=False)
                    else:
                        # hist = (np.histogram(final_photons.loc[:, 'time_rel_line'].values,
                        #                     bins=final_pulses.loc[:, 'time_rel_line'].values)[0])\
                        #     .astype('uint8', copy=False)
                        hist = numba_histogram(final_photons.loc[:, 'time_rel_line'].values,
                                               bins=final_pulses.loc[:, 'time_rel_line'].values)\
                            .astype('uint8', copy=False)
                    finally:
                        if not np.all(hist >= 0):
                            print('WHAT IS GOING ON')
                        assert np.all(hist >= 0), 'In row {}, column {}, the histogram turned out to be negative.'.format(row, col)
                        cur_bincount = numba_bincount(hist).astype('uint64', copy=False)
                        tot_pulses = np.sum(cur_bincount)
                        tot_photons = np.average(cur_bincount, weights=range(len(cur_bincount)))
                        image_bincount[row, col] = BinData(hist=hist, pulses=tot_pulses, photons=tot_photons)
                        all_photons += tot_photons
                        all_pulses += tot_pulses

        return image_bincount, all_pulses, all_photons


@jit((int64[:](uint64[:], uint64[:])), nopython=True, cache=True)
def numba_histogram(arr: np.array, bins) -> np.array:
    return np.histogram(arr, bins)[0]

@jit((int64[:](uint8[:])), nopython=True, cache=True)
def numba_bincount(arr: np.array) -> np.array:
    return np.bincount(arr)
#
# @jit((int64[:](uint64[:], uint64[:])), nopython=True, cache=True)
# def numba_digitize(arr: np.array) -> np.array:
#     pass
