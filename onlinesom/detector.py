import numpy as np
from onlinesom import kohonen
from onlinesom.window import SomData


class SomDetect:
    """
    Use numpy.array
    Given (p)-dim time series
        1. normal data set
        2. online data set
    1. normal SomData - Make normal data-set to SomData
    2. Fit SOM to normal SomData: U-array
    3. online SomData - Make online data-st to SomData
    3. Foreach row (0-axis) of online SomData:
        distance from each U-array
        compare with kohonen.sigma (radius)
        if every value is larger than threshold, anomaly
    """

    def __init__(
            self, path_normal, path_online, cols, window_size, jump_size,
            xdim, ydim, topo = "rectangular", neighbor = "gaussian", dist = "frobenius"
    ):
        """
        :param path_normal: file path of normal data set
        :param path_online: file path of online data set
        :param cols: column index to read
        :param window_size: window size
        :param jump_size: shift size
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space - rectangular or hexagonal
        :param neighbor: Neighborhood function - gaussian or bubble
        :param dist: Distance function - frobenius, nuclear, or
        """
        self.som_tr = SomData(path_normal, cols, window_size, jump_size)
        self.som_te = SomData(path_online, cols, window_size, jump_size)
        self.som_grid = kohonen(self.som_tr.window_data, xdim, ydim, topo, neighbor, dist)
        # anomaly
        self.label = None
        self.window_anomaly = np.empty(self.som_te.window_data.shape[0])
        self.anomaly = np.empty(self.som_te.n)

    def learn_normal(self, epoch = 100, init_rate = None, init_radius = None):
        """
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        """
        self.som_grid.som(self.som_tr.window_data, epoch, init_rate, init_radius)

    def detect_anomaly(self, label = None, threshold = None):
        """
        :param label: anomaly and normal label list
        :param threshold: threshold for detection
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if len(label) != 2:
            raise ValueError("label should have 2 elements")
        self.label = label
        dist_anomaly = np.asarray([self.dist_uarray(i) for i in range(self.som_te.window_data.shape[0])])
        if threshold is None:
            dist_normal = np.asarray([self.dist_normal(i) for i in range(self.som_tr.window_data.shape[0])])
            threshold = np.quantile(dist_normal, 2 / 3)
        som_anomaly = dist_anomaly > threshold
        self.window_anomaly[som_anomaly] = self.label[0]
        self.window_anomaly[np.logical_not(som_anomaly)] = self.label[1]

    def dist_uarray(self, index):
        """
        :param index: Row index for online data set
        :return: minimum distance between online data set and weight matrix
        """
        dist_wt = np.asarray([self.som_grid.dist_mat(self.som_te.window_data, index, j) for j in range(self.som_grid.net.shape[0])])
        return np.min(dist_wt)

    def dist_normal(self, index):
        """
        :param index: Row index for normal data set
        :return: every distance between normal som matrix and weight matrix
        """
        return np.asarray([self.som_grid.dist_mat(self.som_tr.window_data, index, j) for j in range(self.som_grid.net.shape[0])])

    def label_anomaly(self):
        win_size = self.som_te.window_data.shape[1]
        jump_size = (self.som_te.n - win_size) // (self.som_te.window_data.shape[0] - 1)
        # first assign by normal
        self.anomaly = np.repeat(self.label[1], self.anomaly.shape[0])
        for i in range(self.window_anomaly.shape[0]):
            if self.window_anomaly[i] == self.label[0]:
                for j in range(i * jump_size, i * jump_size + win_size):
                    if self.anomaly[j] != self.label[0]:
                        self.anomaly[j] = self.label[0]

