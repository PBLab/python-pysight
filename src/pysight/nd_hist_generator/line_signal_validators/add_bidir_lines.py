from typing import Dict

import numpy as np
import pandas as pd


def add_bidir_lines(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    For unidirectional scans fake line signals have to be inserted for us to identify forward- and
    back-phase photons.
    """

    length_of_lines = data["Lines"].shape[0]
    new_line_arr = np.zeros(length_of_lines * 2 - 1)
    new_line_arr[::2] = data["Lines"].loc[:, "abs_time"].values
    new_line_arr[1::2] = data["Lines"].loc[:, "abs_time"].rolling(window=2).mean()[1:]

    data["Lines"] = pd.DataFrame(new_line_arr, columns=["abs_time"], dtype="uint64")
    return data
