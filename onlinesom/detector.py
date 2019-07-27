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
        if every value is larger than radius, anomaly
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
        :param dist: Distance function - frobenius, pca, or
        """
        self.som_tr = SomData(path_normal, cols, window_size, jump_size)
        self.som_te = SomData(path_online, cols, window_size, jump_size)
        self.som_grid = kohonen(self.som_tr, xdim, ydim, topo, neighbor, dist)

    def learn_normal(self, epoch = 100, init_rate = None, init_radius = None):
        """
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        """
        self.som_grid.som(self.som_tr.window_data, epoch, init_rate, init_radius)

    def detect_anomaly(self, label = None):
        """
        :param label: anomaly and normal label list
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if label.shape != 2:
            raise ValueError("label should have 2 elements")
        dist_anomaly = [self.dist_uarray(i) for i in range(self.som_te.window_data.shape[0])]
        som_anomaly = dist_anomaly > self.som_grid.sigma
        anomaly_result = np.empty(self.som_te.window_data.shape[0])
        anomaly_result[som_anomaly] = label[0]
        anomaly_result[not som_anomaly] = label[1]
        return anomaly_result

    def dist_uarray(self, index):
        """
        :param index: Row index for online data set
        :return: minimum distance between online data set and weight matrix
        """
        dist_wt = np.asarray([self.som_grid.dist_mat(self.som_te.window_data, index, j) for j in range(self.som_grid.net.shape[0])])
        return np.min(dist_wt)

