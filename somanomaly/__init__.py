import numpy as np
from scipy.spatial import distance


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

    Update weight mi(t + 1) = mi(t) + ⍺(t) * hci(t) [x(t) - mi(t)]
    Neighborhood function hci(t) = h(dist(rc, ri), t)
        rc, ri: location vectors of node c and i
    if Gaussian:
        hci(t) = exp(-dist^2 / (2 * σ^2(t)))
        Radius: σ(t) = σ_0 * exp(-t / ƛ)
        Learning rate: ⍺(t) = ⍺_0 * exp(-t / ƛ)
    """

    def __init__(self, data, xdim, ydim, topo = "rectangular", neighbor = "gaussian", dist = "frobenius", seed = None):
        """
        :param data: 3d array. processed data set for Online SOM Detector
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space - rectangular or hexagonal
        :param neighbor: Neighborhood function - gaussian or bubble
        :param dist: Distance function - frobenius, nuclear, or
        :param seed: Random seed
        """
        np.random.seed(seed = seed)
        self.net_dim = np.array([xdim, ydim])
        self.ncol = data.shape[2]
        self.nrow = data.shape[1]
        # Initialize codebook matrix
        self.init_weight()
        # Topology
        topo_types = ["rectangular", "hexagonal"]
        if topo not in topo_types:
            raise ValueError("Invalid topo. Expected one of: %s" % topo_types)
        self.topo = topo
        self.init_grid()
        self.dist_node()
        # Neighborhood function
        neighbor_types = ["gaussian", "bubble"]
        if neighbor not in neighbor_types:
            raise ValueError("Invalid neighbor. Expected one of: %s" % neighbor_types)
        self.neighbor_func = neighbor
        # Distance function
        dist_type = ["frobenius", "nuclear"]
        if dist not in dist_type:
            raise ValueError("Invalid dist. Expected one of: %s" % dist_type)
        self.dist_func = dist
        # som()
        self.alpha = None
        self.sigma = None
        self.time_constant = None
        # find_bmu()
        self.bmu = None

    def init_weight(self):
        self.net = np.random.rand(self.net_dim[0] * self.net_dim[1], self.nrow, self.ncol)

    def init_grid(self):
        self.pts = np.array(
            np.meshgrid(
                range(self.net_dim[0]),
                range(self.net_dim[1])
            )
        ).reshape(2, np.prod(self.net_dim)).T
        if self.topo == "hexagonal":
            self.pts[:, 0] = self.pts[:, 0] + .5 * (self.pts[:, 0] % 2)
            self.pts[:, 1] = np.sqrt(3) / 2 * self.pts[:, 1]

    def som(self, data, epoch = 100, init_rate = None, init_radius = None):
        """
        :param data: 3d array. processed data set for Online SOM Detector
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        """
        num_obs = data.shape[0]
        obs_id = np.arange(num_obs)
        chose_i = np.empty(1)
        bmu_node = np.empty(1)
        # node_id = None
        hci = None
        # learning rate
        if init_rate is None:
            init_rate = .05
        self.alpha = init_rate
        # radius of neighborhood
        if init_radius is None:
            init_radius = np.quantile(self.dci, q = 2 / 3, axis = None)
        self.sigma = init_radius
        # time constant (lambda)
        self.time_constant = epoch / np.log(self.sigma)
        # BMU pair
        self.find_bmu(data)
        bmu_dist = self.dci[1, :]
        for i in range(epoch):
            chose_i = np.random.choice(obs_id, size = 1)
            # BMU
            bmu_node = self.bmu[chose_i.astype(int)]
            bmu_dist = self.dci[bmu_node.astype(int), :].flatten()
            # decay
            self.sigma = kohonen.decay(init_radius, i + 1, self.time_constant)
            self.alpha = kohonen.decay(init_rate, i + 1, self.time_constant)
            # message - remove later
            print("=============================================================")
            print("epoch: ", i + 1)
            print("learning rate: %.3f" % self.alpha)
            print("BMU radius: %.3f" % self.sigma)
            print("------------------------------")
            # neighboring nodes
            neighbor_neuron = np.argwhere(bmu_dist <= self.sigma).flatten()
            # message - remove later
            print("distance between BMU and node: ", bmu_dist)
            print("neighboring neuron: ", neighbor_neuron)
            print("------------------------------")
            for k in range(neighbor_neuron.shape[0]):
                node_id = neighbor_neuron[k]
                hci = self.neighborhood(bmu_dist[node_id], self.sigma)
                # message - remove later
                print("node: ", node_id)
                print("neighborhood function value: %.3f" % hci)
                # update codebook matrices of neighboring nodes
                self.net[node_id, :, :] += \
                    self.alpha * hci * \
                    (data[chose_i.astype(int), :, :] - self.net[node_id, :, :]).reshape((self.nrow, self.ncol))
                # message - remove later
                print("codebook matrix: \n", self.net[node_id, :, :])
                print("------------------------------")

    def find_bmu(self, data):
        """
        :param data: Processed data set for SOM.
        :return: Best Matching Unit index for each observation index (in order)
        """
        dist_code = np.empty(self.net.shape[0]) # node length
        bmu_id = np.empty(data.shape[0]) # observation length
        for i in range(data.shape[0]):
            dist_code = np.asarray([self.dist_mat(data, i, j) for j in range(self.net.shape[0])])
            bmu_id[i] = np.argmin(dist_code)
        self.bmu = bmu_id

    def dist_mat(self, data, index, node):
        """
        :param data: Processed data set for SOM.
        :param index: Randomly chosen observation id for input matrix among 3d tensor set.
        :param node: node index
        :return: distance between input matrix observation and weight matrix of the node
        """
        if self.dist_func == "frobenius":
            return np.linalg.norm(data[index - 1, :, :] - self.net[node, :, :], "fro")
        elif self.dist_func == "nuclear":
            return np.linalg.norm(data[index - 1, :, :] - self.net[node, :, :], "nuc")

    def dist_node(self):
        """
        :return: distance matrix of SOM neuron
        """
        if self.topo == "hexagonal":
            self.dci = distance.cdist(self.pts, self.pts, "euclidean")
        elif self.topo == "rectangular":
            self.dci = distance.cdist(self.pts, self.pts, "chebyshev")

    @staticmethod
    def decay(init, time, time_constant):
        """
        :param init: initial value
        :param time: t
        :param time_constant: lambda
        :return: decaying value of alpha or sigma
        """
        return init * np.exp(-time / time_constant)

    def neighborhood(self, node_distance, radius):
        """
        :param node_distance: Distance between SOM neurons
        :param radius: Radius of BMU neighborhood
        :return: Neighborhood function hci
        """
        if self.neighbor_func == "gaussian":
            return np.exp(-node_distance ** 2 / (2 * (radius ** 2)))
        elif self.neighbor_func == "bubble":
            if node_distance <= radius:
                return 1.0
            else:
                return 0.0
