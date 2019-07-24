import numpy as np

class kohonen:
    """
    Matrix SOM
    Initialize weight matrix
    For epoch <- 1 to N do
        Choose input matrix observation randomly - i
        For k <- 1 to n_node do
            compute d(input matrix i, weight matrix k)
        end
        Best Matching Unit = winning node = node with the smallest distance
        For k <- 1 to n_node do
            update weight matrix
        end
    end
    """

    def __init__(self, data, xdim, ydim, dist):
        """
        :param data: 3d array. processed data set for Online SOM Detector
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param dist: Distance function
        """
        self.net_dim = np.array([xdim, ydim])
        self.ncol = data.shape[2]
        self.nrow = data.shape[1]
        self.init_weight()
        self.dist_func = dist

    def init_weight(self):
        self.net = np.random.rand(self.net_dim[0] + self.net_dim[1], self.ncol, self.nrow)

    def som(self, data, epoch, init_rate):
        """
        :param data: 3d array. processed data set for Online SOM Detector
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :return:
        """
        num_obs = data.shape[0]
        obs_id = np.arange(num_obs)
        for i in range(1, epoch + 1):

            for k in range(1, self.net_dim[0] * self.net_dim[1] + 1):

    def find_bmu(self, data, index):
        """
        :param data: Processed ddata set for SOM.
        :param index: Randomly chosen observation id for input matrix among 3d tensor set.
        :return: Best Matching Unit node and BMU index
        """
        dist_code = [self.dist_mat(data, index, j - 1) for j in range(1, self.net[0] + 1)]
        bmu_id = np.argmin(dist_code)
        bmu = self.net[bmu_id, :, :]
        return bmu, bmu_id

    def dist_mat(self, data, index, node):
        """
        :param data: Processed ddata set for SOM.
        :param index: Randomly chosen observation id for input matrix among 3d tensor set.
        :return: distance between input matrix observation and weight matrix of the node
        """
        if self.dist_func == "frobenius":
            np.linalg.norm(data[i - 1, :, :] - self.net[node, :, :], "fro")

