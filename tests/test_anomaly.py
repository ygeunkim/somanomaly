import numpy as np
from onlinesom.detector import SomDetect


def detect_example(path_normal = "../data/station1_train.csv", path_on = "../data/station1_test.csv"):
    """
    :param path_normal: normal data set
    :param path_on: online data set
    """
    som_anomaly = SomDetect(path_normal, path_on, range(2, 7), 60, 50, 30, 30)
    print("------------------------------")
    print("normal data set: ", som_anomaly.som_tr.window_data.shape)
    print("online data set: ", som_anomaly.som_te.window_data.shape)
    print("------------------------------")
    som_anomaly.learn_normal(100)
    som_anomaly.detect_anomaly(label = [-1, 1])
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


np.set_printoptions(precision = 3)
detect_example()
