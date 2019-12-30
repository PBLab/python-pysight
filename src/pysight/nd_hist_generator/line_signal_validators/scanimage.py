from itertools import tee
import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import attr
import numba


@attr.s(slots=True)
class ScanImageLineValidator:
    sig_val = attr.ib()  # SignalValidator

    def __getattr__(self, item):
        return getattr(self.sig_val, item)

    def run(self) -> Tuple[Dict, np.uint64]:
        """
        Interpolate SI-specific line signals.

        Returns a dictionary containing the data and the mean difference between subsequent lines
        """
        lines = self.dict_of_data["Lines"].loc[:, "abs_time"].copy()
        (
            lines_mat,
            rel_idx,
            end_of_frames_idx,
            last_idx_of_row,
            rel_idx_non_end_frame,
        ) = self.__calc_line_parameters(lines=lines)
        if lines_mat.shape[0] == 1:  # single frame
            lines = pd.Series(lines_mat[0, : self.num_of_lines], dtype=np.uint64)
            delta = np.uint64(lines.diff().mean())
        else:
            theo_lines, delta = self.__gen_line_model(
                lines=lines, rel_idx=rel_idx, end_of_frames_idx=end_of_frames_idx
            )
            lines_mat = self.__diff_mat_analysis(
                y=theo_lines,
                lines_mat=lines_mat,
                last_idx=last_idx_of_row,
                delta=delta,
                rel_idx=rel_idx_non_end_frame,
            )
            lines = self.__finalize_lines(lines_mat=lines_mat)
        if self.bidir:
            lines = self.sig_val.add_phase_to_bidir_lines(lines=lines)
        self.dict_of_data["Lines"] = pd.DataFrame(
            lines, dtype=np.uint64, columns=["abs_time"]
        )
        return self.dict_of_data, delta

    def __gen_line_model(
        self, lines: pd.Series, rel_idx: np.ndarray, end_of_frames_idx: np.ndarray
    ):
        """ Using linear approximation generate a model for the "correct" line signal """
        rel_idx_first_frame = rel_idx[rel_idx < self.num_of_lines]
        lines_first_frame = lines.iloc[: self.num_of_lines].drop(rel_idx_first_frame)
        # Create a model: y = mx + const
        const = lines_first_frame[0]
        m = np.uint64(lines_first_frame.diff().median())
        x = np.arange(start=0, stop=self.num_of_lines, dtype=np.uint64)
        y = m * x + const
        y = np.tile(y, (end_of_frames_idx.shape[0], 1))
        y[1:, :] -= y[1, 0]
        y = np.concatenate((y, np.zeros_like(y)), axis=1)
        return y, m

    def __diff_mat_analysis(
        self,
        y: np.ndarray,
        lines_mat: np.ndarray,
        last_idx: np.ndarray,
        delta: int,
        rel_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Check for missing/extra lines in the matrix of lines.

        :param np.ndarray diff_mat:
        :param np.ndarray last_idx: Last index of relevant lines in the frame
        """
        diff_mat = np.abs(np.subtract(lines_mat, y, dtype=np.int64))
        missing_vals_rows, missing_vals_cols = np.where(diff_mat > delta / 15)
        frames_to_correct = np.where(last_idx != self.num_of_lines)[0]
        for frame_num in frames_to_correct:
            num_of_missing_lines = last_idx[frame_num] - self.num_of_lines
            if num_of_missing_lines < 0:
                for miss in missing_vals_cols[missing_vals_rows == frame_num]:
                    lines_mat[frame_num, :] = np.concatenate(
                        (
                            lines_mat[frame_num, :miss],
                            np.atleast_1d(y[frame_num, miss]),
                            lines_mat[frame_num, miss:-1],
                        )
                    )
            elif num_of_missing_lines > 0:
                cur_missing_cols = missing_vals_cols[
                    missing_vals_rows == frame_num
                ].copy()
                iters = 0
                while (cur_missing_cols.shape[0] > 1) and (iters < 1000):
                    lines_mat[frame_num, :] = np.concatenate(
                        (
                            lines_mat[frame_num, : cur_missing_cols[0]],
                            np.atleast_1d(y[frame_num, cur_missing_cols[0]]),
                            lines_mat[frame_num, cur_missing_cols[1] :],
                        )
                    )
                    diff_line = np.abs(
                        np.subtract(
                            lines_mat[frame_num, :], y[frame_num, :], dtype=np.int64
                        )
                    )
                    cur_missing_cols = np.where(diff_line > delta / 20)[0]
                    iters += 1
                if iters == 1000:
                    logging.warning(
                        "Line signal was corrupt during at least one frame."
                    )
        return lines_mat

    def __finalize_lines(self, lines_mat: np.ndarray) -> pd.Series:
        """ Add back the start-of-frame times to the lines """
        frames_mat = np.array(self.dict_of_data["Frames"]).reshape(
            (len(self.dict_of_data["Frames"]), 1)
        )
        frames_mat = np.tile(frames_mat, (1, self.num_of_lines))
        lines_mat = lines_mat[:, : self.num_of_lines] + frames_mat
        return pd.Series(lines_mat.ravel())

    def __calc_line_parameters(
        self, lines: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Generate general parameters of the given acquisition """
        change = np.abs(lines.diff().pct_change(periods=1))
        rel_idx = np.where(change > self.change_thresh)[0]
        div, mod = divmod(len(lines), self.num_of_lines)
        if len(rel_idx) == 0 and mod != 0 and div == 0:
            raise UserWarning(
                "Data contained incomplete signals only from the first frame."
            )

        end_of_frames = change > 5
        end_of_frames_idx = np.where(end_of_frames)[0]  # scanimage specific
        rel_idx_non_end_frame = np.where(change <= self.change_thresh)[0]
        lines_mat, last_idx_of_row = iter_end_of_frames(
            lines.to_numpy(), end_of_frames_idx, self.num_of_lines
        )
        start_time_of_frames = lines[end_of_frames_idx].values
        start_time_of_frames = np.concatenate(
            (np.array([0], dtype=np.uint64), start_time_of_frames[:-1])
        ).reshape((end_of_frames_idx.shape[0], 1))
        lines_mat -= start_time_of_frames
        if "Frames" not in self.dict_of_data:
            self.dict_of_data["Frames"] = pd.DataFrame(
                start_time_of_frames.ravel(), columns=["abs_time"]
            )

        for row, last_idx in enumerate(last_idx_of_row):
            lines_mat[row, last_idx:] = 0

        return (
            lines_mat,
            rel_idx,
            end_of_frames_idx,
            last_idx_of_row,
            rel_idx_non_end_frame,
        )

    @staticmethod
    def __pairwise(iterable):
        """From itertools: s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


@numba.jit(nopython=True)
def iter_end_of_frames(lines, end_of_frames_idx, num_of_lines):
    """Iterates over two signals - the start and end of each frame - and returns a
    matrix containing the lines between those two signals.
    """
    lines_mat = np.zeros((len(end_of_frames_idx), num_of_lines * 2), dtype=np.uint64)
    last_idx_of_row = np.zeros((end_of_frames_idx.shape[0]), dtype=np.int32)
    frame_start = 0
    for row, end in enumerate(end_of_frames_idx):
        frame_end = end
        last_idx = frame_end - frame_start
        lines_mat[row, :last_idx] = lines[frame_start:frame_end]
        last_idx_of_row[row] = last_idx
        frame_start = end
    return lines_mat, last_idx_of_row
