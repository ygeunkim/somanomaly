import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SomData:
    """
    Use numpy.array
    Given (p)-dim time series
    1. Slide window of size (w) with shift size (jump)
        => window win_num = (n - w) / jump + 1
    2. Bind every window by row.
    3. 3d array of win_num * w * p
    """

    def __init__(self, path, cols, standardization, window_size, jump_size):
        """
        :param path: file path of time series to train SOM
        :param cols: column index to read
        :param standardization: standadize the data?
        :param window_size: window size
        :param jump_size: shift size
        """
        data = SomData.read_array(path, cols, standardization)
        self.n = data.shape[0]
        win_num = (self.n - window_size) // jump_size + 1
        self.window_data = np.empty((win_num, window_size, data.shape[1]))
        for i in range(win_num):
            self.window_data[i, :, :] = data[range(i * jump_size, i * jump_size + window_size), :]

    @staticmethod
    def read_array(path, cols = None, standardization = False):
        """
        :param path: data path
        :param cols: column index to read
        :param standardization: standardize the data?
        :return: numpy converted from pandas
        """
        if cols is None:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, usecols = cols)
        if standardization:
            scalar = StandardScaler()
            df = scalar.fit_transform(df)
            return df
        return pd.DataFrame.to_numpy(df)
