"""
__author__ = Hagai Hargil
"""
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import attr
from itertools import chain


@attr.s(slots=True)
class MScanLineValidator:
    sig_val = attr.ib()  # SignalValidator

    def __getattr__(self, item):
        return getattr(self.sig_val, item)

    def run(self) -> Tuple[Dict, np.uint64]:
        """
        Interpolate MScan-specific line signals
        :return: Dictionary containing the data and the mean difference between subsequent lines
        """
        lines = self.dict_of_data['Lines'].loc[:, 'abs_time'].copy()
        rel_idx, delta = self.__calc_line_parameters(lines=lines)
        lines, rel_idx, delta = self.__filter_extra_lines(lines=lines, delta=delta)
        if len(rel_idx) > 0:  # missing lines, not just extra
            theo_lines = self.__gen_line_model(lines=lines, m=delta)
            lines = self.__diff_vec_analysis(lines=lines, y=theo_lines, delta=delta)
        lines = self.__finalize_lines(lines=lines, delta=delta)
        if self.bidir:
            lines = self.sig_val.add_phase_to_bidir_lines(lines=lines)

        self.dict_of_data['Lines'] = pd.DataFrame(lines, dtype=np.uint64, columns=['abs_time'])
        return self.dict_of_data, delta

    def __calc_line_parameters(self, lines: pd.Series) -> Tuple[np.ndarray, np.uint64]:
        """ Generate general parameters of the given acquisition """
        rel_idx = np.where(np.abs(lines.diff().pct_change(periods=1)) > self.change_thresh)[0]
        delta = np.uint64(lines.drop(rel_idx).reindex(np.arange(len(lines))).interpolate().diff().mean())
        return rel_idx[::2], delta

    def __filter_extra_lines(self, lines: pd.Series, delta: np.uint64) -> Tuple[pd.Series, pd.Series, np.uint64]:
        """
        Kick out excess line signals
        :param lines:
        :param delta:
        :return: Tuple of valid lines, missing lines and new delta of lines
        """
        diffs = lines.diff()
        rel_idx = np.where(np.abs(diffs.pct_change(periods=1)) > self.change_thresh)[0]
        recurring = np.where(np.diff(rel_idx) == 1)[0]
        idx_to_keep = np.ones_like(rel_idx, dtype=bool)

        for idx, _ in enumerate(recurring[1:], 1):
            try:
                if recurring[idx] - recurring[idx-1] == 1:
                    idx_to_keep[recurring[idx] + 1] = False
            except IndexError:
                pass

        rel_idx_new = pd.Series(rel_idx[idx_to_keep][::2], dtype=np.uint64)
        missing_lines = []
        extra_lines = []

        for idx in rel_idx_new:
            if diffs[idx] < (delta/2):  # excess lines
                extra_lines.append(idx)
            else:
                missing_lines.append(idx)

        valid_lines = lines.drop(extra_lines).reset_index(drop=True)
        delta = np.uint64(valid_lines.drop(missing_lines).diff().mean())

        # Get rid of lines that came after the last frame
        num_of_extra_lines = len(valid_lines) % self.num_of_lines
        valid_lines = valid_lines[:-num_of_extra_lines]

        return valid_lines, pd.Series(missing_lines), delta

    def __gen_line_model(self, lines: pd.Series, m: np.uint64) -> np.ndarray:
        """ Using linear approximation generate a model for the "correct" line signal """
        const = lines.iloc[0]
        x = np.arange(start=0, stop=len(lines), dtype=np.uint64)
        y = m * x + const
        if len(lines) > 1500:  # correct simulated lines
            idx_range = np.arange(1500, len(lines), step=1500, dtype=np.uint64)
            for idx in idx_range:
                x = np.arange(0, len(lines)-idx, dtype=np.uint64)
                y[idx:] = m * x + lines.iloc[idx]

        # MScan's lines are evenly separated
        first_diff = np.uint64(lines.iloc[2:110].diff()[1::2].median())
        delta_diff = np.int32((m - first_diff) / 2)
        if first_diff < m:
            y[1::2] -= delta_diff
            y[::2] += delta_diff
        else:
            y[1::2] += delta_diff
            y[::2] -= delta_diff
        return y

    def __diff_vec_analysis(self, y: np.ndarray, lines: pd.Series,
                                  delta: np.uint64) -> pd.Series:
        diff_vec = np.abs(np.subtract(y, lines, dtype=np.int64))
        missing_val = np.where(diff_vec > delta / 20)[0]
        while missing_val.shape[0] > 0:
            if np.abs(diff_vec[missing_val[0]] - delta)/delta < 0.1:  # double line
                lines = np.concatenate((lines[:missing_val[0]],
                                        lines[missing_val[0]+1:]))
            else:
                lines = np.concatenate((lines[:missing_val[0]],
                                        np.atleast_1d(y[missing_val[0]]),
                                        lines[missing_val[0]:]))
            # Restart the loop
            y = self.__gen_line_model(pd.Series(lines), delta)
            diff_vec = np.abs(np.subtract(y, lines, dtype=np.int64))
            missing_val = np.where(diff_vec > delta / 20)[0]
        return pd.Series(lines)

    def __finalize_lines(self, lines, delta) -> pd.Series:
        """ Sample the lines so that they're "silent" between frames """
        lines_between_frames = int(np.rint(self.frame_delay / delta))
        start_of_frame_idx = np.arange(start=0, stop=len(lines),
                                       step=self.num_of_lines + lines_between_frames,
                                       dtype=np.uint64)
        end_of_frame_idx = start_of_frame_idx + self.num_of_lines
        exact_lines = [lines[slice(start, end)] for start, end in zip(start_of_frame_idx, end_of_frame_idx)]
        exact_lines = np.array(list(chain.from_iterable(exact_lines)), dtype=np.uint64)
        self.sig_val.num_of_frames = len(end_of_frame_idx)

        return pd.Series(exact_lines, dtype=np.uint64)
