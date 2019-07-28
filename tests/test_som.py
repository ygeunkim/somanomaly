import numpy as np
from onlinesom import kohonen
from onlinesom.window import SomData


def train_example(path = "../data/station1_train.csv"):
    """
    :param path: normal data set path
    """
    win_data = SomData(path, range(2, 7), 300, 50)
    print("------------------------------")
    print(win_data.window_data.shape)
    print("------------------------------")
    som_grid = kohonen(win_data.window_data, 10, 10)
    som_grid.som(data = win_data.window_data)


np.set_printoptions(precision = 3)
train_example()
