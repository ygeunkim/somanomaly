import sys
import numpy as np
from somanomaly import kohonen
from somanomaly.window import SomData


def train_example(path):
    """
    :param path: normal data set path
    """
    win_data = SomData(path, range(2, 7), 300, 50)
    print("------------------------------")
    print(win_data.window_data.shape)
    print("------------------------------")
    som_grid = kohonen(win_data.window_data, 10, 10)
    som_grid.som(data = win_data.window_data)
    som_grid.plot_error()
    som_grid.plot_heatmap(win_data.window_data)


np.set_printoptions(precision = 3)
if __name__ == "__main__":
    path = sys.argv[1]
    train_example(path)
