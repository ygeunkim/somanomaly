import numpy as np
import pandas as pd


class SomData:
    """
    Use numpy.array
    Given (p)-dim time series
    1. Slide window of size (w) with shift size (jump)
        => window win_num = (n - w) / jump + 1
    2. Bind every window by row.
    3. 3d array of win_num * w * p
    """

    def __init__(self, path, cols, window_size, jump_size):
        """
        :param path: file path of time series to train SOM
        :param cols: column index to read
        :param window_size: window size
        :param jump_size: shift size
        """
        data = SomData.read_array(path, cols)
        win_num = (data.shape[0] - window_size) // jump_size + 1
        self.window_data = np.empty((win_num, window_size, jump_size))
        for i in range(win_num):
            self.window_data[i, :, :] = data[i * jump_size + i * jump_size + window_size - 1, :]

    @staticmethod
    def read_array(path, cols):
        """
        :param path: data path
        :param cols: column index to read
        :return: numpy converted from pandas
        """
        df = pd.read_csv(path, index_col = cols)
        return pd.Series.to_numpy(df)
