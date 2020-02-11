import numpy as np
import pandas as pd
import argparse
import time
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import f
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from sklearn.metrics import classification_report
from somanomaly import kohonen
from somanomaly.window import SomData


class SomDetect:
    """
    Use numpy.array
    Given (p)-dim time series
        1. normal data set
        2. online data set
    1. normal SomData - Make normal data-set to SomData
    2. Fit SOM to normal SomData: U-array
    3. online SomData - Make online data-set to SomData
    4. Foreach row (0-axis) of online SomData:
        distance from each U-array
        compare with threshold
        if every value is larger than threshold, anomaly
    """

    def __init__(
            self, path_normal, path_online, cols = None, standard = False,
            window_size = 60, jump_size = 60, test_log = False,
            xdim = None, ydim = None, topo = "rectangular", neighbor = "gaussian",
            dist = "frobenius", decay = "exponential", seed = None
    ):
        """
        :param path_normal: file path of normal data set
        :param path_online: file path of online data set
        :param cols: column index to read
        :param standard: standardize both data sets
        :param window_size: window size
        :param jump_size: shift size
        :param test_log: log-scale normal series
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space - rectangular or hexagonal
        :param neighbor: Neighborhood function - gaussian or bubble
        :param dist: Distance function - frobenius, nuclear, or
        :param decay: decaying learning rate and radius - exponential or linear
        :param seed: Random seed
        """
        self.som_tr = SomData(path_normal, cols, window_size, jump_size, test_log)
        self.som_te = SomData(path_online, cols, window_size, jump_size, False)
        self.som_grid = kohonen(self.som_tr.window_data, xdim, ydim, topo, neighbor, dist, decay, seed)
        self.win_size = window_size
        self.jump = jump_size
        # standardization
        self.standard = standard
        if self.standard:
            scaler = StandardScaler()
            # standardize normal data-set
            tmp_tr = self.som_tr.window_data.reshape((-1, self.som_tr.window_data.shape[2]))
            tmp_tr = scaler.fit_transform(tmp_tr).reshape(self.som_tr.window_data.shape)
            self.som_tr.window_data = tmp_tr
            # standardize online data-set
            tmp_te = self.som_te.window_data.reshape((-1, self.som_te.window_data.shape[2]))
            tmp_te = scaler.fit_transform(tmp_te).reshape(self.som_te.window_data.shape)
            self.som_te.window_data = tmp_te
        # anomaly
        self.label = None
        self.window_anomaly = np.empty(self.som_te.window_data.shape[0])
        self.anomaly = np.empty(self.som_te.n)
        # som settings
        self.topo = topo
        self.h = neighbor
        self.d = dist
        self.decay = decay
        # plot
        self.project = None

    def learn_normal(self, epoch = 100, init_rate = None, init_radius = None):
        """
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        """
        self.som_grid.som(self.som_tr.window_data, epoch, init_rate, init_radius, keep_net = True)
        self.net = self.som_grid.net

    def detect_anomaly(
            self, label = None,
            level = .9, clt_test = "gai", mfdr = None, power = None,
            clt_map = False, neighbor = None
    ):
        """
        :param label: anomaly and normal label list
        :param level: what quantile to use. Change this for ztest, clt, or cltlind threshold
        :param clt_test: what multiple testing method to use for clt and cltlind - bh, invest, gai, or lord
        :param mfdr: eta of alpha-investing
        :param power: rho of GAI
        :param clt_map: use mapped codebook for clt and cltlind?
        :param neighbor: radius - use neighboring nodes when clt_map
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if len(label) != 2:
            raise ValueError("label should have 2 elements")
        self.label = label
        test_types = ["bon", "bh", "invest", "gai", "lord"]
        if clt_test not in test_types:
            raise ValueError("Invalid clt_test. Expected on of: %s" % test_types)
        som_anomaly = None
        # SomAnomaly procedure
        if clt_map:
            if self.som_grid.project is None:
                normal_distance = np.asarray(
                    [self.som_grid.dist_weight(self.som_tr.window_data, i) for i in
                     tqdm(range(self.som_tr.window_data.shape[0]), desc="mapping")]
                )
                self.som_grid.dist_normal = normal_distance[:, 0]
                self.som_grid.project = normal_distance[:, 1]
            # mapped nodes
            normal_project, proj_count = np.unique(self.som_grid.project, return_counts=True)
            # neighboring nodes
            if neighbor is not None:
                proj_dist = np.argwhere(
                    self.som_grid.dci[normal_project.astype(int), :] <= neighbor
                )
                normal_project, neighbor_count = np.unique(proj_dist[:, 0], return_counts=True)
                proj_count = np.repeat(proj_count, neighbor_count)
                normal_project = np.unique(proj_dist[:, 1])
            # corresponding codebook
            net_stand = self.net[normal_project.astype(int), :, :]
        else:
            net_stand = self.net
            proj_count = np.repeat(1, self.net.shape[0])
        normal_distance = np.asarray(
            [self.dist_normal(net_stand, j) for j in tqdm(range(net_stand.shape[0]), desc="pseudo-population")]
        )
        # online set
        dist_anomaly = np.asarray(
            [self.dist_codebook(net_stand, k, w=proj_count) for k in
             tqdm(range(self.som_te.window_data.shape[0]), desc="codebook distance")]
        )
        # cltlind
        clt_mean = np.average(normal_distance[:, 0], weights=proj_count)
        sn = np.sqrt(
            np.sum(normal_distance[:, 1] * proj_count)
        )
        # sn = np.sqrt(
        #     net_stand.shape[0] * np.average(normal_distance[:, 1], weights = proj_count)
        # )
        # not iid - lindeberg clt sn = sqrt(sum(sigma2 ** 2)) => sum(xi - mui) / sn -> N(0, 1)
        self.dstat = net_stand.shape[0] * (dist_anomaly - clt_mean) / sn
        # pvalue
        pvalue = 1 - norm.cdf(self.dstat)
        # multiple test
        if clt_test == "bon":
            alpha = 1 - level
            # pj <= alpha * 2^(-j)
            som_anomaly = pvalue <= alpha * pow(.5, np.arange(self.som_te.window_data.shape[0]) + 1)
        elif clt_test == "bh":
            alpha = 1 - level
            alpha /= self.som_te.window_data.shape[0]
            # Benjamini–Hochberg - i * alpha / N for ordered p-value
            alpha *= np.arange(self.som_te.window_data.shape[0]) + 1
            # Benjamini–Hochberg 1 find the largest k - p(k) <= alpha(k)
            # 2 reject every H(j), j = 1, ..., k
            p_ordered = np.argsort(pvalue)
            test = np.argwhere(pvalue[p_ordered] <= alpha)
            som_anomaly = np.full(self.som_te.window_data.shape[0], fill_value=False, dtype=bool)
            if test.shape[0] != 0:
                test_k = np.max(test)
                som_anomaly[p_ordered[:(test_k + 1)]] = True
        elif clt_test == "invest":
            alpha = 1 - level
            if mfdr is None:
                mfdr = 1 - alpha
            # alpha-investing
            wealth = alpha * mfdr
            som_anomaly = True
            k = 0
            for j in tqdm(range(self.som_te.window_data.shape[0]), desc="alpha-investing"):
                h0 = j + 1
                alphaj = wealth / (1 + h0 - k)
                rj = pvalue[j] <= alphaj  # boolean - 1 if reject, 0 if accept
                som_anomaly = np.append(som_anomaly, rj)
                wealth += - (1 - rj) * alphaj / (1 - alphaj) + rj * alpha
                # wealth += (1 - rj) * np.log(1 - alphaj) + rj * (alpha + np.log(1 - pvalue[j]))
                k = (1 - rj) * k + rj * h0  # the most recently rejected hypothesis
            som_anomaly = som_anomaly[1:]
        elif clt_test == "gai":
            alpha = 1 - level
            # upper bound on the power
            if power is None:
                power = 1
            if mfdr is None:
                mfdr = 1 - alpha
            # W(0)
            wealth = alpha * mfdr
            som_anomaly = True
            for j in tqdm(range(self.som_te.window_data.shape[0]), desc="gai"):
                # phi(k) = w(k - 1) / 10, k = 1, 2, ...
                # relative scheme - until w(j) < w(0) / 1000
                # relative200 scheme - until 200 tests
                phi = wealth / 10
                # alpha(k) s.t. phi(k) / rho(k) = phi(k) / alpha(k) - 1
                # rho(k) = 1 => alpha(k) = phi(k) / (phi(k) + 1)
                alphaj = (phi * power) / (phi + power)
                rj = pvalue[j] <= alphaj
                som_anomaly = np.append(som_anomaly, rj)
                # psi(k) = min(phi(k) / rho(k) + alpha, phi(k) / alpha(k) + alpha - 1)
                psi = np.minimum(phi / power + alpha, phi / alphaj + alpha - 1)
                # w(k) = w(k - 1) - phi(k) + R(k)psi(k)
                wealth += -phi + rj * psi
            som_anomaly = som_anomaly[1:]
        elif clt_test == "lord":
            alpha = 1 - level
            if mfdr is None:
                mfdr = 1 - alpha
            som_anomaly = True
            # W(0)
            wealth = np.array([alpha * mfdr])
            # last discovery time
            tau = 0
            # gamma seq of LORD3
            gamma = []
            for j in tqdm(range(self.som_te.window_data.shape[0]), desc="lord"):
                gamma = np.append(
                    gamma,
                    .0772 * np.log(np.maximum(j + 1, 2)) / ((j + 1) * np.exp(np.sqrt(np.log(j + 1))))
                )
                # alpha(j) = gamma(j - tau) * w(tau)
                alphaj = gamma[j - tau] * wealth[tau]
                rj = pvalue[j] <= alphaj
                som_anomaly = np.append(som_anomaly, rj)
                tau = rj * j
                # w(j) = w(j - 1) - alpha(j) + R(j) * alpha
                wealth = np.append(
                    wealth,
                    wealth[j] - alphaj + rj * alpha
                )
            som_anomaly = som_anomaly[1:]
        # label
        self.window_anomaly[som_anomaly] = self.label[0]
        self.window_anomaly[np.logical_not(som_anomaly)] = self.label[1]

    def dist_uarray(self, index):
        """
        :param index: Row index for online data set
        :return: minimum distance between online data set and weight matrix
        """
        dist_wt = np.asarray([self.som_grid.dist_mat(self.som_te.window_data, index, j) for j in tqdm(range(self.net.shape[0]), desc = "bmu")])
        return np.min(dist_wt), np.argmin(dist_wt)

    def dist_codebook(self, codebook, index, w = None):
        """
        :param codebook: transformed codebook matrices
        :param index: Row index for online data set
        :param w: Weight for average
        :return: average distance between online data set and weight matrix
        """
        net_num = codebook.shape[0]
        dist_wt = np.asarray(
            [self.dist_mat(self.som_te.window_data[index, :, :], codebook[i, :, :]) for i in tqdm(range(net_num), desc = "averaging")]
        )
        if w is None:
            w = np.repeat(1, net_num)
        return np.average(dist_wt, weights = w)

    def dist_normal(self, codebook, node):
        """
        :param codebook: transformed codebook matrices
        :param node: node index
        :return: average and sd of distances between normal som matrix and chosen weight matrix
        """
        dist_wt = np.asarray(
            [self.dist_mat(codebook[node, :, :], self.som_tr.window_data[i, :, :]) for i in tqdm(range(self.som_tr.window_data.shape[0]), desc = "mean and sd")]
        )
        return np.average(dist_wt), np.var(dist_wt)

    def dist_dij(self, codebook, index):
        """
        :param codebook: transformed codebook matrices
        :param index: Row index for online data set
        :return: every distance pair between online data set and weight matrix
        """
        net_num = codebook.shape[0]
        dist_wt = np.asarray(
            [self.dist_mat(self.som_te.window_data[index, :, :], codebook[i, :, :]) for i in tqdm(range(net_num), desc = "averaging")]
        )
        return dist_wt

    def dist_mat(self, mat1, mat2):
        """
        :param mat1: Matrix
        :param mat2: Matrix
        :return: distance between mat1 and mat2
        """
        x = mat1 - mat2
        if self.som_grid.dist_func == "frobenius":
            return np.linalg.norm(x, "fro")
        elif self.som_grid.dist_func == "nuclear":
            return np.linalg.norm(x, "nuc")
        elif self.som_grid.dist_func == "mahalanobis":
            covmat = np.cov(x, rowvar = False)
            w, v = np.linalg.eigh(covmat)
            w[w == 0] += .0001
            covinv = v.dot(np.diag(1 / w)).dot(v.T)
            ss = x.dot(covinv).dot(x.T)
            return np.sqrt(np.trace(ss))
        elif self.som_grid.dist_func == "eros":
            covmat = np.cov(x, rowvar = False)
            u, s, vh = randomized_svd(covmat, n_components = covmat.shape[1], n_iter = 1, random_state = None)
            w = s / s.sum()
            ss = np.multiply(vh, w).dot(vh.T)
            return np.sqrt(np.trace(ss))

    def label_anomaly(self):
        win_size = self.win_size
        jump_size = self.jump
        # first assign by normal
        self.anomaly = np.repeat(self.label[1], self.anomaly.shape[0])
        for i in tqdm(range(self.window_anomaly.shape[0]), desc = "anomaly unit change"):
            if self.window_anomaly[i] == self.label[0]:
                self.anomaly[(i * jump_size):(i * jump_size + win_size)] = self.label[0]

    def plot_heatmap(self):
        """
        :return: heatmap of projection onto normal SOM
        """
        if self.project is None:
            som_dist_calc = np.asarray(
                [self.dist_uarray(i) for i in tqdm(range(self.som_te.window_data.shape[0]), desc="mapping online set")]
            )
            self.project = som_dist_calc[:, 1]
        xdim = self.som_grid.net_dim[0]
        x = self.project % xdim
        y = self.project // xdim
        if self.som_grid.topo == "rectangular":
            fig = go.Figure(
                go.Histogram2d(
                    x = x,
                    y = y,
                    colorscale="Viridis"
                )
            )
            fig.show()
        elif self.som_grid.topo == "hexagonal":
            x = x + .5 * (y % 2)
            y = np.sqrt(3) / 2 * y
            # plt_hex = plt.hexbin(x, y)
            # plt.close()
            # fig = tls.mpl_to_plotly(plt_hex)
            plt.hexbin(x, y)
            plt.show()

    @classmethod
    def detect_block(cls, normal_list, online_list, col_list, standard = False,
                     window_size = 60, jump_size = 60, test_log = False,
                     xdim = None, ydim = None, topo = "rectangular", neighbor = "gaussian",
                     dist = "frobenius", decay = "exponential", seed = None,
                     epoch = 100, init_rate = None, init_radius = None
                     label = None, level = .9, clt_test = "gai", mfdr = None, power = None):
        """
        :param normal_list: normal series files
        :param online_list: online series files
        :param col_list: column range array for each file
        :param standard: standardize both data sets
        :param window_size: window size
        :param jump_size: shift size
        :param test_log: log-scale streaming series
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space - rectangular or hexagonal
        :param neighbor: Neighborhood function - gaussian or bubble
        :param dist: Distance function - frobenius, nuclear, or
        :param decay: decaying learning rate and radius - exponential or linear
        :param seed: Random seed
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        :param label: anomaly and normal label list
        :param level: what quantile to use. Change this for ztest, clt, or cltlind threshold
        :param clt_test: what multiple testing method to use for clt and cltlind - bh, invest, gai, or lord
        :param mfdr: eta of alpha-investing
        :param power: rho of GAI
        :return: Anomaly detection
        """
        num_tr = len(normal_list)
        num_te = len(online_list)
        if num_tr != num_te:
            raise ValueError("Invalid file list. normal_list and online_list should be same length.")
        anomaly_df = None
        for b in tqdm(range(num_tr), desc = "block"):
            som_anomaly = cls(
                normal_list[b], online_list[b], range(col_list[b, 0], col_list[b, 1]),
                standard, window_size, jump_size, test_log, xdim, ydim, topo, neighbor, dist, decay, seed
            )
            som_anomaly.learn_normal(epoch, init_rate, init_radius)
            som_anomaly.detect_anomaly(label, level = level, clt_test = clt_test,
                                       mfdr = mfdr, power = power,
                                       clt_map = True, neighbor = None)
            col_name = ".pred" + str(b)
            if b == 0:
                anomaly_df = pd.DataFrame({col_name: som_anomaly.window_anomaly})
            else:
                anomaly_df[col_name] = som_anomaly.anomaly
            anomaly = anomaly_df.any(axis = 1).to_numpy()
            anomaly_label = np.repeat(label[1], som_anomaly.anomaly.shape[0])
            for i in range(som_anomaly.window_anomaly.shape[0]):
                if anomaly[i] == label[0]:
                    anomaly_label[(i * jump_size):(i * jump_size + window_size)] = label[0]
            return anomaly_label

def main():
    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument(
        "normal",
        type = str,
        help = "Normal dataset file"
    )
    parser.add_argument(
        "online",
        type = str,
        help = "Streaming dataset file"
    )
    parser.add_argument(
        "output",
        type = str,
        help = "Output"
    )
    parser.add_argument(
        "-c", "--column",
        type = str,
        help = "Column index to read - start,end"
    )
    parser.add_argument(
        "-e", "--eval",
        help = "True label dataset file. if specified, give the traditional precision/recall"
    )
    parser.add_argument(
        "--log",
        help = "Log transform",
        action = "store_true"
    )
    # SOM training
    parser.add_argument(
        "--standardize",
        help = "Standardize both data sets",
        action = "store_true"
    )
    parser.add_argument(
        "-w", "--window",
        type = int,
        default = 30,
        help = "Window size (Default = 30)"
    )
    parser.add_argument(
        "-j", "--jump",
        type = int,
        default = 30,
        help = "Shift size (Default = 30)"
    )
    parser.add_argument(
        "-x", "--xgrid",
        type = int,
        default = None,
        help = "Number of x-grid (Default = sqrt(N * sqrt(N)))"
    )
    parser.add_argument(
        "-y", "--ygrid",
        type = int,
        default = None,
        help = "Number of y-grid (Default = sqrt(N * sqrt(N)))"
    )
    parser.add_argument(
        "-p", "--prototype",
        type = str,
        default = "hexagonal",
        help = "Topology of SOM output space - hexagonal (default) or rectangular"
    )
    parser.add_argument(
        "-n", "--neighborhood",
        type = str,
        default = "gaussian",
        help = "Neighborhood function - gaussian (default), triangular, or bubble"
    )
    parser.add_argument(
        "-m", "--metric",
        type = str,
        default = "frobenius",
        help = "Distance function - frobenius (default), nuclear, mahalanobis, or eros"
    )
    parser.add_argument(
        "-d", "--decay",
        type = str,
        default = "linear",
        help = "Decaying function - linear (default) or exponential"
    )
    parser.add_argument(
        "-s", "--seed",
        type = int,
        help = "Random seed (Default = system time)"
    )
    parser.add_argument(
        "-i", "--iter",
        type = int,
        default = 50,
        help = "Epoch number (Default = 50)"
    )
    parser.add_argument(
        "-a", "--alpha",
        type = float,
        default = .1,
        help = "Initial learning rate (Default = 0.1)"
    )
    parser.add_argument(
        "-r", "--radius",
        type = float,
        help = "Initial radius of BMU neighborhood (Default = 2/3 quantile of every distance between nodes)"
    )
    # Anomaly detection
    parser.add_argument(
        "-l", "--label",
        type = str,
        default = "1,0",
        help = "Anomaly and normal labels, e.g. 1,0 (default)"
    )
    parser.add_argument(
        "-u", "--threshold",
        type = str,
        default = "cltlind",
        help = "Threshold method - cltlind (default), clt, anova, ztest, mean, quantile, radius, inv_som, kmeans, hclust"
    )
    # clt and cltlind

    parser.add_argument(
        "-o", "--overfit",
        help = "Use only mapped codebook",
        action = "store_true"
    )
    parser.add_argument(
        "--find",
        type = float,
        help = "When using mapped codebook, their neighboring nodes also can be used. - radius for neighbor"
    )
    parser.add_argument(
        "-q", "--multiple",
        type = str,
        default = "gai",
        help = "Multiple testing method - gai (default), invest, lord, or bh"
    )
    # Plot
    parser.add_argument(
        "-1", "--error",
        help = "Plot reconstruction error for each epoch",
        action = "store_true"
    )
    parser.add_argument(
        "-2", "--heat",
        help = "Plot heatmap of SOM",
        action = "store_true"
    )
    parser.add_argument(
        "-3", "--pred",
        help = "Plot heatmap of projection onto normal SOM",
        action = "store_true"
    )
    # assign arguments
    args = parser.parse_args()
    normal_file = args.normal
    online_file = args.online
    normal_list = None
    online_list = None
    if str(args.normal).strip().find(",") != -1:
        normal_list = str(args.normal).strip().split(",")
    if str(args.online).strip().find(",") != -1:
        online_list = str(args.online).strip().split(",")
    output_list = None
    dstat_file = None
    if str(args.output).strip().find(",") != -1:
        output_list = str(args.output).strip().split(",")
        output_file = output_list[0]
        dstat_file = output_list[1]
    else:
        output_file = args.output
    cols = None
    col_list = None
    if args.column is not None:
        if str(args.column).count(",") == 1:
            cols = str(args.column).strip().split(",")
            cols = range(int(cols[0]), int(cols[1]))
        elif str(args.column).count(",") > 1:
            col_tmp = str(args.column).strip().split(",")
            col_list = np.array(col_tmp).astype(int).reshape((-1, 2))
    if args.eval is not None:
        print_eval = True
        true_file = args.eval
        target_names = ["anomaly", "normal"]
    test_log = args.log
    standard = args.standardize
    window_size = args.window
    jump_size = args.jump
    xdim = args.xgrid
    ydim = args.ygrid
    topo = args.prototype
    neighbor = args.neighborhood
    dist = args.metric
    decay = args.decay
    seed = args.seed
    epoch = args.iter
    init_rate = args.alpha
    init_radius = args.radius
    label = str(args.label).strip().split(",")
    label = [int(label[0]), int(label[1])]
    threshold_list = None
    ztest_opt = .9
    if str(args.threshold).strip().find(",") != -1:
        threshold_list = str(args.threshold).strip().split(",")
        threshold = threshold_list[0]
        ztest_opt = float(threshold_list[1])
    else:
        threshold = args.threshold
    proj = args.overfit
    neighbor_node = args.find
    multiple_list = None
    eta = None
    rho = None
    if str(args.multiple).strip().find(",") != -1:
        multiple_list = str(args.multiple).strip().split(",")
        multiple_test = multiple_list[0]
        if str(multiple_list[1]).strip().find("+") != -1:
            multiple_opt = str(multiple_list[1]).strip().split("+")
            eta = float(multiple_opt[0])
            rho = float(multiple_opt[1])
        else:
            eta = float(multiple_list[1])
    elif str(args.multiple).strip().find("+") != -1:
        multiple_list = str(args.multiple).strip().split("+")
        multiple_test = multiple_list[0]
        rho = float(multiple_list[1])
    else:
        multiple_test = args.multiple
    print_error = args.error
    print_heat = args.heat
    print_projection = args.pred
    # somanomaly
    start_time = time.time()
    if normal_list is None:
        som_anomaly = SomDetect(normal_file, online_file, cols, standard,
                                window_size, jump_size, test_log,
                                xdim, ydim, topo, neighbor, dist, decay, seed)
        som_anomaly.learn_normal(epoch = epoch, init_rate = init_rate, init_radius = init_radius)
        som_anomaly.detect_anomaly(label = label, threshold = threshold,
                                   level = ztest_opt, clt_test = multiple_test, mfdr = eta, power = rho,
                                   clt_map = proj, neighbor = neighbor_node)
        som_anomaly.label_anomaly()
        anomaly_df = pd.DataFrame({".pred": som_anomaly.anomaly})
        anomaly_df.to_csv(output_file, index = False, header = False)
        if dstat_file is not None:
            dstat_df = pd.DataFrame({".som": som_anomaly.dstat})
            dstat_df.to_csv(dstat_file, index = False, header = False)
            window_df = pd.DataFrame({".pred": som_anomaly.window_anomaly})
            window_df.to_csv(dstat_file.replace(".csv", "_pred.csv"), index = False, header = False)
    else:
        anomaly_pred = SomDetect.detect_block(
            normal_list, online_list, col_list,
            standard, window_size, jump_size, test_log,
            xdim, ydim, topo, neighbor, dist, decay, seed,
            epoch, init_rate, init_radius, label, ztest_opt, multiple_test, eta, rho,
        )
        anomaly_df = pd.DataFrame({".pred": anomaly_pred})
        anomaly_df.to_csv(output_file, index = False, header = False)
    print("")
    print("process for %.2f seconds================================================\n" %(time.time() - start_time))
    # files
    print("Files-------------------------------------")
    if normal_list is None:
        print("Normal data: ", normal_file)
        print("Streaming data: ", online_file)
        print("Anomaly detection: ", output_file)
        if dstat_file is not None:
            print("SomAnomly statistic: ", dstat_file)
            print("Window prediction: ", dstat_file.replace(".csv", "_pred.csv"))
        # print parameter
        print("SOM parameters----------------------------")
        if som_anomaly.standard:
            print("Standardized!")
        print("Initial learning rate: ", som_anomaly.som_grid.initial_learn)
        print("Initial radius: ", som_anomaly.som_grid.initial_r)
        print("[Window, jump]: ", [som_anomaly.win_size, som_anomaly.jump])
        print("SOM grid: ", som_anomaly.som_grid.net_dim)
        print("Topology: ", som_anomaly.som_grid.topo)
        print("Neighborhood function: ", som_anomaly.som_grid.neighbor_func)
        print("Decay function: ", som_anomaly.som_grid.decay_func)
        print("Distance function: ", som_anomaly.som_grid.dist_func)
        print("Epoch number: ", som_anomaly.som_grid.epoch)
    else:
        print("Normal data: ", normal_list)
        print("Streaming data: ", online_list)
        print("Anomaly detection: ", output_file)
        # print parameter
        print("SOM parameters----------------------------")
        print("Initial learning rate: ", init_rate)
        print("Initial radius: ", init_radius)
        print("[Window, jump]: ", [window_size, jump_size])
        print("SOM grid: ", np.array([xdim, ydim]))
        print("Topology: ", topo)
        print("Neighborhood function: ", neighbor)
        print("Decay function: ", decay)
        print("Distance function: ", dist)
        print("Epoch number: ", epoch)
    print("------------------------------------------")
    if threshold_list is not None:
        print("Anomaly detection by %s of %.3f" %(threshold, ztest_opt))
    else:
        print("Anomaly detection by ", threshold)
    if threshold == "clt" or threshold == "cltlind":
        if multiple_list is not None:
            if eta is not None:
                if rho is not None:
                    print("with multiple testing %s of %.3f and %.3f" % (multiple_test, eta, rho))
                else:
                    print("with multiple testing %s of %.3f" %(multiple_test, eta))
        else:
            print("with multiple testing %s" % multiple_test)
    print("==========================================")
    # evaluation
    if normal_list is None:
        if print_eval:
            true_anomaly = pd.read_csv(true_file, header = None)
            true_anomaly = pd.DataFrame.to_numpy(true_anomaly)
            print(
                classification_report(
                    true_anomaly, som_anomaly.anomaly,
                    labels = label, target_names = target_names
                )
            )
        # plot
        if print_error or print_heat or print_projection:
            plot_start = time.time()
            if print_error:
                som_anomaly.som_grid.plot_error()
            if print_heat:
                som_anomaly.som_grid.plot_heatmap(som_anomaly.som_tr.window_data)
            if print_projection:
                som_anomaly.plot_heatmap()
            print("Plotting time: %.2f seconds" % (time.time() - plot_start))
    else:
        if print_eval:
            true_anomaly = pd.read_csv(true_file, header=None)
            true_anomaly = pd.DataFrame.to_numpy(true_anomaly)
            print(
                classification_report(
                    true_anomaly, anomaly_pred,
                    labels=label, target_names=target_names
                )
            )


if __name__ == '__main__':
    np.set_printoptions(precision = 3)
    main()
