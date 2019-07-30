import sys
import numpy as np
import pandas as pd
from somanomaly.detector import SomDetect


def detect_save(path_normal, path_on, col_id = range(2, 51), win = 60, jump = 50, xgrid = 20, ygrid = 20):
    """
    :param path_normal: file path of normal data-set
    :param path_on: file path of online data-set
    :param col_id: column index
    :param win: window size
    :param jump: shift size
    :param xgrid: number of x-grid
    :param ygrid: number of y-grid
    :return: save csv file for detected anomaly labels
    """
    som_anomaly = SomDetect(path_normal, path_on, col_id,
                            window_size = win, jump_size = jump,
                            xdim = xgrid, ydim = ygrid)
    som_anomaly.learn_normal(100)
    som_anomaly.detect_anomaly(label = [1, 0], threshold = "mean")
    som_anomaly.label_anomaly()
    anomaly_df = pd.DataFrame(som_anomaly.anomaly)
    anomaly_df.to_csv("../data/processed/som_label_mean.csv", index = False)


np.set_printoptions(precision = 3)
if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    col_from = int(sys.argv[3])
    col_to = int(sys.argv[4])
    win = int(sys.argv[5])
    jump = int(sys.argv[6])
    xgrid = int(sys.argv[7])
    ygrid = int(sys.argv[8])
    detect_save(path_normal = path1, path_on = path2,
                col_id = range(col_from, col_to), win = win, jump = jump, xgrid = xgrid, ygrid = ygrid)

