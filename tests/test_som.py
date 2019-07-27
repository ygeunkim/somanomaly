from onlinesom import kohonen
from onlinesom.window import SomData


def train_example(path = "../data/station1_train.csv"):
    """
    :param path: normal data set path
    :return:
    """
    win_data = SomData(path, range(2, 7), 300, 10)
    som_grid = kohonen(win_data.window_data, 20, 20)
    som_grid.som(data = win_data.window_data)
    return som_grid.net

train_example()