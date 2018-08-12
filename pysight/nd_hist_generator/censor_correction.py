"""
__author__ = Hagai Hargil
"""
import attr
import numpy as np
import pandas as pd
from attr.validators import instance_of
from pysight.nd_hist_generator.movie import Movie, FrameChunk
from collections import deque, namedtuple
from typing import Tuple, Union
from numba import jit, uint8, int64


@attr.s(slots=True)
class CensorCorrection(object):
    raw              = attr.ib(validator=instance_of(dict))
    data             = attr.ib(validator=instance_of(pd.DataFrame))
    movie            = attr.ib(validator=instance_of(Movie))
    all_laser_pulses = attr.ib()
    nano_flim_list   = attr.ib(init=False)
    flim             = attr.ib(default=False, validator=instance_of(bool))
    reprate          = attr.ib(default=80e6, validator=instance_of(float))
    binwidth         = attr.ib(default=800e-12, validator=instance_of(float))
    laser_offset     = attr.ib(default=3.5, validator=instance_of(float))
    num_of_channels  = attr.ib(default=1, validator=instance_of(int))

    @property
    def bins_bet_pulses(self) -> int:
        return int(np.ceil(1 / (self.reprate * self.binwidth)))

    @property
    def offset(self):
        return int(np.floor(self.laser_offset * 10**-9 / self.binwidth))

    def run(self):
        """
        Main pipeline for the censor correction part.
        """
        if self.flim:
            print("Starting the censor correction...")
            self.create_arr_of_hists_deque()
        else:
            print("FLIM deactivated, no censor correction performed.")

    def __gen_laser_pulses_deque(self) -> np.ndarray:
        """
        If data has laser pulses - return them. Else - simulate them with an offset
        """
        start_time = 0
        step = self.bins_bet_pulses
        volumes_in_movie = self.movie.gen_of_volumes()

        if self.all_laser_pulses == 0 and self.flim == False:  # no 'Laser' data was recorded
            for vol in volumes_in_movie:
                yield np.arange(start=start_time+self.offset, stop=vol.end_time,
                                step=step, dtype=np.uint64)

        elif self.all_laser_pulses == 0 and self.flim == True:
            pass

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

    def __worker_arr_of_hists(self, vol):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      binwidth=self.binwidth, reprate=self.reprate)
            return censored.gen_arr_of_hists()

    def create_arr_of_hists_deque(self):
        """
        For each volume generate a single matrix with the same size as the underlying volume,
        which contains a histogram of photons in their laser pulses for each pixel.
        :return: deque() that contains an array of histograms in each place
        """
        self.nano_flim_list = []  # each cell contains a different data channel
        for chan in range(1, self.num_of_channels + 1):
            print("Starting channel number {}: ".format(chan))
            volumes_in_movie = self.movie.gen_of_volumes(channel_num=chan)
            self.nano_flim_list.append([self.__worker_arr_of_hists(vol) for vol in volumes_in_movie])

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

    def append_laser_line(self):
        """
        Add a final laser line to the laser signal input.
        """
        last_laser_row = pd.DataFrame({'abs_time': self.raw['Laser']['abs_time'].iat[-1] + self.bins_bet_pulses,
                                       'edge': 0,
                                       'sweep': self.raw['Laser']['sweep'].iat[-1],
                                       'time_rel_sweep': self.raw['Laser']['time_rel_sweep'].iat[-1] + self.bins_bet_pulses},
                                      index=[self.raw['Laser'].shape[0]])
        self.raw['Laser'] = pd.concat([self.raw['Laser'], last_laser_row])

    def train_dataset(self):
        """
        Using almost raw data, allocate photons to their laser pulses
        (instead of laser pulses to photons) and create all 16 bit words for the ML algorithm.
        :return:
        """
        # Append a fake laser pulse to retain original number of "bins"
        self.append_laser_line()
        sorted_indices = pd.cut(self.raw['PMT1']['abs_time'], bins=self.raw['Laser']['abs_time'],
                                labels=self.raw['Laser'].iloc[:-1, 3], include_lowest=True)
        self.raw['Laser'].set_index(keys='time_rel_sweep', inplace=True,
                                    append=True, drop=True)
        num_of_pos_bins = 22
        new_bins = np.arange(-10, num_of_pos_bins + 1)  # 32 bins
        min_time_after_sweep = 10
        max_time_after_sweep = self.raw['Laser']['time_rel_sweep'].max() - num_of_pos_bins - 1
        indices = np.arange(min_time_after_sweep, max_time_after_sweep)
        hist_df = pd.DataFrame([], dtype=object)
        for idx in indices:
            cur_pulse = self.raw['Laser'].xs(idx, level='time_rel_sweep',
                                             drop_level=False)


@attr.s(slots=True)
class CensoredVolume(object):
    df           = attr.ib(validator=instance_of(pd.DataFrame))
    chunk        = attr.ib(validator=instance_of(FrameChunk))
    offset       = attr.ib(validator=instance_of(int))
    binwidth     = attr.ib(default=800e-12)
    reprate      = attr.ib(default=80e6)

    @property
    def bins_bet_pulses(self) -> int:
        return int(np.ceil(1 / (self.reprate * self.binwidth)))

    def gen_bincount(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin the photons into their relative laser pulses, and count how many photons arrived due to each pulse.
        """
        hist, _ = np.histogram(self.chunk.data.loc[:, 'time_rel_frames'].values, bins=self.laser_pulses)
        dig = np.digitize(self.chunk.data.loc[:, 'time_rel_frames'].values,
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
        sorted_indices: np.ndarray = np.searchsorted(pulses, self.movie.data['time_rel_frames'].values) - 1
        array_of_laser_starts = pulses[sorted_indices]
        subtracted_times = self.movie.data['time_rel_frames'].values - array_of_laser_starts
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

        hist, edges = self.chunk.create_hist()
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
        self.chunk.data.loc[:, 'bins_x'] = (np.digitize(self.chunk.data.loc[:, 'time_rel_frames'].values,
                                          bins=edges[0]) - 1).astype('uint16', copy=False)
        self.chunk.data.loc[:, 'bins_y'] = (np.digitize(self.chunk.data.loc[:, 'time_rel_line'].values,
                                          bins=edges[1]) - 1).astype('uint16', copy=False)
        pulses.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)
        self.chunk.data.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)

        # Go through each bin and histogram the photons there
        image_bincount = np.zeros_like(hist, dtype=object)
        for row in range(self.chunk.x_pixels):
            row_pulses = pulses.xs(key=row, level='bins_x', drop_level=False)
            assert len(row_pulses) > 0, 'Row {} contains no pulses'.format(row)
            try:
                row_photons = self.chunk.data.xs(key=row, level='bins_x', drop_level=False)
            except KeyError:  # no photons in row
                for col in range(self.chunk.y_pixels):
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
                for col in range(self.movie.y_pixels):
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

    def gen_arr_of_hists(self) -> np.ndarray:
        """
        For each specific frame, sort the photons inside the pixels to gqin
        statistics on the distribution of the photons inside the pixels.
        :return: np.ndarray of the same size as the original frame,
        with each bin containing a histogram.
        """
        edges = self.chunk._Volume__create_hist_edges()[0]

        hist_t, edges_t = self.chunk.create_hist()
        assert "time_rel_pulse" in self.chunk.data.columns, \
            "No `time_rel_pulse` column in data."

        self.chunk.data.loc[:, 'bins_x'] = (np.digitize(self.chunk.data.loc[:, 'time_rel_frames'].values,
                                                      bins=edges[0])-1).astype('uint16', copy=False)
        self.chunk.data.loc[:, 'bins_y'] = (np.digitize(self.chunk.data.loc[:, 'time_rel_line'].values,
                                                      bins=edges[1])-1).astype('uint16', copy=False)
        self.chunk.data.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)
        self.chunk.data.drop(len(edges[0])-1, level='bins_x', inplace=True)
        self.chunk.data.drop(len(edges[1])-1, level='bins_y', inplace=True)

        image_bincount = np.zeros((len(edges[0])-1, len(edges[1])-1), dtype=object)  # returned variable, contains hists
        active_lines = np.unique(self.chunk.data.index.get_level_values('bins_x'))
        for row in active_lines:
            print("Row number: {}".format(row))
            row_photons = self.chunk.data.xs(key=row, level='bins_x', drop_level=False)
            rel_idx = np.unique(row_photons.index.get_level_values('bins_y'))
            image_bincount[row, rel_idx] = self.__allocate_photons_to_bins(idx=rel_idx, photons=row_photons)

        return image_bincount

    def __allocate_photons_to_bins(self, idx: np.ndarray, photons: pd.DataFrame):
        """
        Generate a summed-histogram of photons for the relevant edges.
        :param photons:
        :param edge_x:
        :param edge_y:
        :return:
        """
        hist_storage = np.zeros_like(idx, dtype=object)
        for cur_idx, col in enumerate(idx):
            cur_photons = photons.xs(key=col, level='bins_y', drop_level=False)
            hist_storage[cur_idx] = numba_histogram(cur_photons.loc[:, 'time_rel_pulse'].values,
                                                    bins=np.arange(0, self.bins_bet_pulses + 1, dtype=np.uint8))\
                                        .astype('uint8', copy=False)
        return hist_storage


@jit((int64[:](uint8[:], uint8[:])), nopython=True, cache=True)
def numba_histogram(arr: np.array, bins) -> np.array:
    return np.histogram(arr, bins)[0]

@jit((int64[:](uint8[:])), nopython=True, cache=True)
def numba_bincount(arr: np.array) -> np.array:
    return np.bincount(arr)
#
# @jit((int64[:](uint64[:], uint64[:])), nopython=True, cache=True)
# def numba_digitize(arr: np.array) -> np.array:
#     pass
