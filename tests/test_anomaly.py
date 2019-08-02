import sys
import numpy as np
from somanomaly.detector import SomDetect


def detect_example(path_normal, path_on, win = 60, jump = 50, xgrid = 20, ygrid = 20):
    """
    :param path_normal: normal data set
    :param path_on: online data set
    :param win: window size
    :param jump: shift size
    :param xgrid: number of x-grid
    :param ygrid: number of y-grid
    """
    som_anomaly = SomDetect(path_normal, path_on, range(2, 51), standardization = True,
                            window_size = win, jump_size = jump,
                            xdim = xgrid, ydim = ygrid)
    print("------------------------------")
    print("normal data set: ", som_anomaly.som_tr.window_data.shape)
    print("online data set: ", som_anomaly.som_te.window_data.shape)
    print("------------------------------")
    som_anomaly.learn_normal(100)
    som_anomaly.detect_anomaly(label = [1, 0], threshold = "radius")
    count = np.unique(som_anomaly.window_anomaly, return_counts = True)
    print("=============================================================")
    print("window anomaly result: ", som_anomaly.window_anomaly)
    print("counts: ", count)
    print("------------------------------")
    som_anomaly.label_anomaly()
    ent_count = np.unique(som_anomaly.anomaly, return_counts = True)
    print("------------------------------")
    print("anomaly result: ", som_anomaly.anomaly)
    print("counts: ", ent_count)
    print("------------------------------")
    som_anomaly.plot_heatmap()


np.set_printoptions(precision = 3)
if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    detect_example(path1, path2)
