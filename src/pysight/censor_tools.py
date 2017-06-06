"""
__author__ = Hagai Hargil
"""
import attr
import numpy as np
import pandas as pd
from attr.validators import instance_of
from pysight.movie_tools import Volume
from collections import deque, namedtuple


@attr.s(slots=True)
class CensorCorrection(object):
    df = attr.ib(validator=instance_of(pd.DataFrame))
    movie = attr.ib()
    reprate = attr.ib(validator=instance_of(float))
    binwidth = attr.ib(validator=instance_of(float))
    offset = attr.ib(validator=instance_of(int))
    all_laser_pulses = attr.ib()

    def gen_laser_pulses_deque(self) -> np.ndarray:
        """
        If data has laser pulses - return them. Else - simulate them with an offset
        """
        laser_pulses_deque = deque()
        pulse_grid = namedtuple('PulseGrid', ('x_pulses', 'y_pulses'))
        start_time = 0
        step = int(np.ceil(1 / (self.reprate * self.binwidth)))
        volumes_in_movie = self.movie.gen_of_volumes()

        if self.all_laser_pulses == 0:  # no 'Laser' data was recorded
            for vol in volumes_in_movie:
                x_pulses = np.arange(start=start_time+self.offset, stop=vol.end_time,
                                     step=step, dtype=np.uint64)
                y_pulses = np.arange(start=start_time+self.offset, stop=vol.metadata['Y'].end+step,
                                     step=step)
                grid = pulse_grid(x_pulses, y_pulses)
                yield grid
        else:
            for vol in volumes_in_movie:
                x_pulses = self.all_laser_pulses[(self.all_laser_pulses >= vol.abs_start_time-step) &
                                                   (self.all_laser_pulses <= vol.end_time+step)] + self.offset
                y_pulses = 1
                grid = pulse_grid(x_pulses, y_pulses)
                yield grid

    def get_bincount_deque(self):
        bincount_deque = deque()
        laser_pulses_deque = self.gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque).x_pulses)
            bincount_deque.append(censored.gen_bincount())
        return bincount_deque

    def find_temporal_structure_deque(self):
        temp_struct_deque = deque()
        laser_pulses_deque = self.gen_laser_pulses_deque()
        volumes_in_movie = self.movie.gen_of_volumes()
        for idx, vol in enumerate(volumes_in_movie):
            censored = CensoredVolume(df=vol.data, vol=vol, offset=self.offset,
                                      laser_pulses=next(laser_pulses_deque).x_pulses,
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
                                      laser_pulses=next(laser_pulses_deque).x_pulses,
                                      binwidth=self.binwidth, reprate=self.reprate)
            temp_struct_deque.append(censored.gen_array_of_hists())
        return temp_struct_deque


    def generate_label_for_dataset(self) -> float:
        """
        For a given, fixed, power of the laser, find the ratio of photons per pulse
        :return: float
        """
        deque_of_vols = self.create_array_of_hists_deque()


    def learn_histograms(self):
        from sklearn import svm, metrics
        import matplotlib.pyplot as plt

        # Flatten the array
        data = self.gen_array_of_hists_deque().flatten()
        n_samples = len(data)
        labels = self.gen_labels()
        classifier = svm.SVC(gamma=0.001)
        classifier.fit(data[:n_samples / 2], labels[:n_samples / 2])

        # Predictions
        expected = labels[n_samples / 2:]
        predicted = classifier.predict(data[n_samples / 2:])
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

    def gen_bincount(self) -> np.ndarray:
        """
        Bin the photons into their relative laser pulses, and count how many photons arrived due to each pulse.
        """
        hist, _ = np.histogram(self.df['abs_time'].values, bins=self.laser_pulses)
        return np.bincount(hist)

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
        sorted_indices: np.ndarray = np.searchsorted(pulses, self.df['abs_time'].values) - 1
        array_of_laser_starts = pulses[sorted_indices]
        subtracted_times = self.df['abs_time'].values - array_of_laser_starts
        return subtracted_times, array_of_laser_starts, sorted_indices

    def gen_array_of_hists(self) -> np.ndarray:
        """
        For a specific frame, sort photons and laser pulses inside the pixels to gain
        statistics on the distribution of the photons inside the pixels.
        :return: np.ndarray of the same size as the original image. Each pixels contains
        a histogram inside it.
        """
        hist, edges = self.vol.create_hist()

        # Create a relative timestamp to the line signal for each laser pulse
        lines_in_vol = self.df.index.get_level_values('Lines').categories.values
        sorted_pulses = np.searchsorted(lines_in_vol, self.laser_pulses) - 1
        pulses = pd.DataFrame(data=self.laser_pulses[np.where(sorted_pulses >= 0)[0]], columns=['abs_time'])
        pulses = pulses.assign(Lines=lines_in_vol[sorted_pulses[np.where(sorted_pulses >= 0)[0]]])
        pulses.dropna(how='any', inplace=True)
        pulses.loc[:, 'Lines'] = pulses.loc[:, 'Lines'].astype('uint64')
        pulses.loc[:, 'time_rel_line'] = pulses.loc[:, 'abs_time'] - pulses.loc[:, 'Lines']
        pulses.loc[:, 'Lines'] = pulses.loc[:, 'Lines'].astype('category')
        pulses.set_index(keys=['Lines'], inplace=True, append=True, drop=True)

        # Allocate laser pulses and photons to their bins
        pulses.loc[:, 'bins_x']  = (np.digitize(pulses.loc[:, 'abs_time'].values,
                                                bins=edges[0]) - 1).astype('uint16', copy=False)
        pulses.loc[:, 'bins_y']  = (np.digitize(pulses.loc[:, 'time_rel_line'].values,
                                                bins=edges[1]) - 1).astype('uint16', copy=False)
        self.df.loc[:, 'bins_x'] = (np.digitize(self.df.loc[:, 'abs_time'].values,
                                                bins=edges[0]) - 1).astype('uint16', copy=False)
        self.df.loc[:, 'bins_y'] = (np.digitize(self.df.loc[:, 'time_rel_line'].values,
                                                bins=edges[1]) - 1).astype('uint16', copy=False)
        pulses.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)
        self.df.set_index(keys=['bins_x', 'bins_y'], inplace=True, append=True, drop=True)

        # Go through each bin and histogram the photons there
        image_bincount = np.zeros_like(hist, dtype=object)
        for row in range(self.vol.x_pixels):
            row_pulses = pulses.xs(key=row, level='bins_x', drop_level=False)
            assert len(row_pulses) > 0, 'Row {} contains no pulses'.format(row)
            try:
                row_photons = self.df.xs(key=row, level='bins_x', drop_level=False)
            except KeyError:  # no photons in row
                for col in range(self.vol.y_pixels):
                    final_pulses = row_pulses.xs(key=col, level='bins_y', drop_level=False)
                    hist = (np.histogram(np.array([]), bins=final_pulses.loc[:, 'time_rel_line'].values)[0])\
                           .astype('uint8')
                    image_bincount[row, col] = np.bincount(hist).astype('uint64', copy=False)
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
                        image_bincount[row, col] = np.bincount(hist).astype('uint64', copy=False)

        return image_bincount


