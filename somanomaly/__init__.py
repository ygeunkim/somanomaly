import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy.spatial import distance
from tqdm import tqdm


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

    def __init__(
            self, data, xdim, ydim, topo = "rectangular", neighbor = "gaussian",
            dist = "frobenius", decay = "exponential", seed = None
    ):
        """
        :param data: 3d array. processed data set for Online SOM Detector
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space - rectangular or hexagonal
        :param neighbor: Neighborhood function - gaussian, bubble, or triangular
        :param dist: Distance function - frobenius, nuclear, mahalanobis (just form of mahalanobis), or
        :param decay: decaying learning rate and radius - exponential or linear
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
        neighbor_types = ["gaussian", "bubble", "triangular"]
        if neighbor not in neighbor_types:
            raise ValueError("Invalid neighbor. Expected one of: %s" % neighbor_types)
        self.neighbor_func = neighbor
        # Distance function
        dist_type = ["frobenius", "nuclear", "mahalanobis"]
        if dist not in dist_type:
            raise ValueError("Invalid dist. Expected one of: %s" % dist_type)
        self.dist_func = dist
        # Decay
        decay_types = ["exponential", "linear"]
        if decay not in decay_types:
            raise ValueError("Invalid decay. Expected one of: %s" % decay_types)
        self.decay_func = decay
        # som()
        self.epoch = None
        self.alpha = None
        self.sigma = None
        self.initial_learn = None
        self.initial_r = None
        self.time_constant = None
        # find_bmu()
        self.bmu = None
        # plot
        self.reconstruction_error = None
        self.dist_normal = None
        self.project = None

    def init_weight(self):
        self.net = np.random.rand(self.net_dim[0] * self.net_dim[1], self.nrow, self.ncol)

    def init_grid(self):
        """
        [row_pts, col_pts]
        xdim x ydim rows (points)
        [1,1]
        [2,1]
        [1,2]
        [2,2]
        2--------->
        1--------->^
        """
        self.pts = np.array(
            np.meshgrid(
                np.arange(self.net_dim[0]) + 1,
                np.arange(self.net_dim[1]) + 1
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
        node_id = None
        hci = None
        self.epoch = epoch
        # learning rate
        if init_rate is None:
            init_rate = .5
        self.alpha = init_rate
        self.initial_learn = init_rate
        # radius of neighborhood
        if init_radius is None:
            init_radius = np.quantile(self.dci, q = 2 / 3, axis = None)
        self.sigma = init_radius
        self.initial_r = init_radius
        # time constant (lambda)
        if self.decay_func == "exponential":
            self.time_constant = epoch / np.log(self.sigma)
        elif self.decay_func == "linear":
            self.time_constant = 1 / epoch
        # distance between nodes
        bmu_dist = self.dci[1, :]
        rcst_err = np.empty(epoch)
        for i in tqdm(range(epoch), desc = "epoch"):
            chose_i = int(np.random.choice(obs_id, size = 1))
            # BMU - self.bmu
            self.find_bmu(data, chose_i)
            # reconstruction error - sum of distances from BMU
            rcst_err[i] = np.sum([np.square(self.dist_mat(data, j, self.bmu.astype(int))) for j in range(data.shape[0])])
            bmu_dist = self.dci[self.bmu.astype(int), :].flatten()
            # decay
            self.sigma = self.decay(init_radius, i + 1, self.time_constant)
            self.alpha = self.decay(init_rate, i + 1, self.time_constant)
            # neighboring nodes (includes BMU)
            neighbor_neuron = np.argwhere(bmu_dist <= self.sigma).flatten()
            for k in tqdm(range(neighbor_neuron.shape[0]), desc = "updating"):
                node_id = neighbor_neuron[k]
                hci = self.neighborhood(bmu_dist[node_id], self.sigma)
                # update codebook matrices of neighboring nodes
                self.net[node_id, :, :] += \
                    self.alpha * hci * \
                    (data[chose_i, :, :] - self.net[node_id, :, :]).reshape((self.nrow, self.ncol))
        self.reconstruction_error = pd.DataFrame({"Epoch": np.arange(self.epoch) + 1, "Reconstruction Error": rcst_err})
        # Map to SOM = BMU
        # normal_distance = np.asarray([self.dist_weight(data, i) for i in tqdm(range(data.shape[0]), desc = "mapping")])
        # self.dist_normal = normal_distance[:, 0]
        # self.project = normal_distance[:, 1]

    def find_bmu(self, data, index):
        """
        :param data: Processed data set for SOM.
        :param index: Randomly chosen observation id for input matrix among 3d tensor set.
        """
        dist_code = np.asarray([self.dist_mat(data, index, j) for j in range(self.net.shape[0])])
        self.bmu = np.argmin(dist_code)

    def dist_mat(self, data, index, node):
        """
        :param data: Processed data set for SOM.
        :param index: Randomly chosen observation id for input matrix among 3d tensor set.
        :param node: node index
        :return: distance between input matrix observation and weight matrix of the node
        """
        if self.dist_func == "frobenius":
            return np.linalg.norm(data[index, :, :] - self.net[node, :, :], "fro")
        elif self.dist_func == "nuclear":
            return np.linalg.norm(data[index, :, :] - self.net[node, :, :], "nuc")
        elif self.dist_func == "mahalanobis":
            x = data[index, :, :] - self.net[node, :, :]
            covmat = np.cov(x.T)
            ss = x.dot(np.linalg.inv(covmat)).dot(x.T)
            return np.sqrt(np.trace(ss))

    def dist_node(self):
        """
        :return: distance matrix of SOM neuron
        """
        if self.topo == "hexagonal":
            self.dci = distance.cdist(self.pts, self.pts, "euclidean")
        elif self.topo == "rectangular":
            self.dci = distance.cdist(self.pts, self.pts, "chebyshev")

    def decay(self, init, time, time_constant):
        """
        :param init: initial value
        :param time: t
        :param time_constant: lambda
        :return: decaying value of alpha or sigma
        """
        if self.decay_func == "exponential":
            return init * np.exp(-time / time_constant)
        elif self.decay_func == "linear":
            return init - time * time_constant

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
        elif self.neighbor_func == "triangular":
            if node_distance <= radius:
                return 1 - np.abs(node_distance) / radius
            else:
                return 0.0

    def dist_weight(self, data, index):
        """
        :param data: Processed data set for SOM
        :param index: index for data
        :return: minimum distance between input matrix and weight matrices, its node id (BMU)
        """
        dist_wt = np.asarray([self.dist_mat(data, index, j) for j in tqdm(range(self.net.shape[0]), desc = "bmu")])
        return np.min(dist_wt), np.argmin(dist_wt)

    def plot_error(self):
        """
        :return: line plot of reconstruction error versus epoch
        """
        fig = px.line(self.reconstruction_error, x = "Epoch", y = "Reconstruction Error")
        fig.show()

    def plot_heatmap(self, data):
        """
        :return: Heatmap for SOM
        """
        if self.project is None:
            normal_distance = np.asarray(
                [self.dist_weight(data, i) for i in tqdm(range(data.shape[0]), desc="mapping")]
            )
            self.dist_normal = normal_distance[:, 0]
            self.project = normal_distance[:, 1]
        x = self.project % self.net_dim[0]
        y = self.project // self.net_dim[0]
        fig = go.Figure(
            go.Histogram2d(
                x = x,
                y = y,
                colorscale = "Viridis"
            )
        )
        fig.show()
