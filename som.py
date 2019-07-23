import numpy as np

class kohonen:
    """
    Matrix SOM
    Initialize weight matrix
    For epoch <- 1 to N do
        Choose input matrix randomly
        For k <- 1 to n_node do
            compute d(input matrix, weight matrix)
        end
        Best Matching Unit = winning node = node with the smallest distance
        For k <- 1 to n_node do
            update weight matrix
        end
    end
    """

    def __init__(self, xdim, ydim, win_size, num_vars):
        """
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param win_size: Window size
        :param num_vars: Number of variables - nrow of input matrix
        """
        self.net_dim = np.array([xdim, ydim])
        self.win_size = win_size
        self.num_vars = num_vars
        self.init_weight()

    def init_weight(self):
        self.net = np.random.rand(self.net_dim[0], self.net_dim[1], self.win_size, self.num_vars)

    def som(self, data, epoch, init_rate):
        """
        :param data: 3d array. processed data set for Online SOM Detector
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :return:
        """
        nrow = data.shape[0]
        for i in range(1, epoch + 1):
            for k in range(1, self.net_dim[0] * self.net_dim[1] + 1):
