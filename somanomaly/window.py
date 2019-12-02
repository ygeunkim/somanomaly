import numpy as np
import pandas as pd
from tqdm import tqdm


class SomData:
    """
    Use numpy.array
    Given (p)-dim time series
    1. Slide window of size (w) with shift size (jump)
        => window win_num = (n - w) / jump + 1
    2. Bind every window by row.
    3. 3d array of win_num * w * p
    """

    def __init__(self, path, cols, window_size, jump_size, log_scale = False):
        """
        :param path: file path of time series to train SOM
        :param cols: column index to read
        :param window_size: window size
        :param jump_size: shift size
        :param log_scale: log scaled data
        """
        data = SomData.read_array(path, cols)
        self.n = data.shape[0]
        win_num = (self.n - window_size) // jump_size + 1
        self.window_data = np.empty((win_num, window_size, data.shape[1]))
        for i in tqdm(range(win_num), desc = "bind window"):
            if log_scale:
                self.window_data[i, :, :] = np.log(data[range(i * jump_size, i * jump_size + window_size), :] + 1)
            else:
                self.window_data[i, :, :] = data[range(i * jump_size, i * jump_size + window_size), :]

    @staticmethod
    def read_array(path, cols = None):
        """
        :param path: data path
        :param cols: column index to read
        :return: numpy converted from pandas
        """
        if cols is None:
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, usecols = cols)
        return pd.DataFrame.to_numpy(df)
