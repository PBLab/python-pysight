"""
__author__ = Hagai Hargil
"""
import attr
import numpy as np
import pandas as pd
from attr.validators import instance_of
from pysight.movie_tools import Volume, Movie
from collections import deque, namedtuple
from typing import Tuple


@attr.s(slots=True)
class CensorCorrection(object):
    movie = attr.ib(validator=instance_of(Movie))
    reprate = attr.ib(validator=instance_of(float))
    binwidth = attr.ib(validator=instance_of(float))
    offset = attr.ib(validator=instance_of(int))
    power = attr.ib(validator=instance_of(int))
    all_laser_pulses = attr.ib()

    @property
    def bins_bet_pulses(self) -> int:
        return int(np.ceil(1 / (self.reprate * self.binwidth)))

    def gen_laser_pulses_deque(self) -> np.ndarray:
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

    def get_bincount_deque(self):
        print("Movie object created. Generating the bincount deque...")
        bincount_deque = deque()
        laser_pulses_deque = self.gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque))
            dig, bincount = censored.gen_bincount()
            photon_hist = np.zeros((len(dig), self.bins_bet_pulses), dtype=np.uint8)
            for laser_idx, photon in enumerate(np.nditer(censored.df.loc[:, 'time_rel_frames'].values)):
                start_time = censored.laser_pulses[dig[laser_idx]]
                try:
                    end_time = censored.laser_pulses[dig[laser_idx] + 1]
                except IndexError:  # photons out of laser pulses
                    continue
                else:
                    photon_hist[laser_idx, :] = np.histogram(photon, bins=np.arange(start_time, end_time + 1,
                                                                                 dtype='uint64'))[0].tolist()
            data_dict = {'photon_hist'    : photon_hist,
                         'bincount'       : bincount,
                         'num_empty_hists': sum(bincount) - laser_idx}
            bincount_deque.append(data_dict)

        return bincount_deque

    def find_temporal_structure_deque(self):
        temp_struct_deque = deque()
        laser_pulses_deque = self.gen_laser_pulses_deque()
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
        laser_pulses_deque = self.gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque),
                                      binwidth=self.binwidth, reprate=self.reprate)
            temp_struct_deque.append(censored.gen_array_of_hists())
        return temp_struct_deque


    def gen_labels(self, size) -> np.ndarray:
        """
        Create labels for the ML algorithm. Label value must be an integer.
        :size: Number of elements
        :return: np.ndarray
        """
        return np.ones(size, dtype=np.uint8) * self.power

    def learn_histograms(self, bincount):
        from sklearn import svm, metrics
        import matplotlib.pyplot as plt

        print("Bincount done. Adding all data to a single matrix.")
        data = np.empty((0, self.bins_bet_pulses))
        for vol in bincount:
            data = np.r_[data, vol['photon_hist']]  # the histograms with photons in them
            data = np.r_[data, np.zeros((vol['num_empty_hists'], self.bins_bet_pulses),
                                        dtype=np.uint8)]  # empty hists
        n_samples = data.shape[0]
        labels = self.gen_labels(n_samples)
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


@attr.s(slots=True)
class CensoredVolume(object):
    df = attr.ib(validator=instance_of(pd.DataFrame))
    vol = attr.ib(validator=instance_of(Volume))
    laser_pulses = attr.ib(validator=instance_of(np.ndarray))
    offset = attr.ib(validator=instance_of(int))
    binwidth = attr.ib(default=800e-12)
    reprate = attr.ib(default=80e6)

    def gen_bincount(self) -> Tuple[np.ndarray]:
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
        pulses.loc[:, 'bins_x']  = (np.digitize(pulses.loc[:, 'time_rel_frames'].values,
                                                bins=edges[0]) - 1).astype('uint16', copy=False)
        pulses.loc[:, 'bins_y']  = (np.digitize(pulses.loc[:, 'time_rel_line'].values,
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
                        hist = (np.histogram(final_photons.loc[:, 'time_rel_line'].values,
                                            bins=final_pulses.loc[:, 'time_rel_line'].values)[0])\
                               .astype('uint8', copy=False)
                    finally:
                        if not np.all(hist >= 0):
                            print('WHAT IS GOING ON')
                        assert np.all(hist >= 0), 'In row {}, column {}, the histogram turned out to be negative.'.format(row, col)
                        cur_bincount = np.bincount(hist).astype('uint64', copy=False)
                        tot_pulses = np.sum(cur_bincount)
                        tot_photons = np.average(cur_bincount, weights=range(len(cur_bincount)))
                        image_bincount[row, col] = BinData(hist=hist, pulses=tot_pulses, photons=tot_photons)
                        all_photons += tot_photons
                        all_pulses += tot_pulses

        return image_bincount, all_pulses, all_photons


