import numpy as np
import pandas as pd
import sys
import getopt
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
            window_size = 60, jump_size = 60,
            xdim = 20, ydim = 20, topo = "rectangular", neighbor = "gaussian",
            dist = "frobenius", decay = "exponential", seed = None
    ):
        """
        :param path_normal: file path of normal data set
        :param path_online: file path of online data set
        :param cols: column index to read
        :param standard: standardize both data sets
        :param window_size: window size
        :param jump_size: shift size
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space - rectangular or hexagonal
        :param neighbor: Neighborhood function - gaussian or bubble
        :param dist: Distance function - frobenius, nuclear, or
        :param decay: decaying learning rate and radius - exponential or linear
        :param seed: Random seed
        """
        self.som_tr = SomData(path_normal, cols, window_size, jump_size)
        self.som_te = SomData(path_online, cols, window_size, jump_size)
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

    def learn_normal(self, epoch = 100, init_rate = None, init_radius = None, subset_net = 100):
        """
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        :param subset_net: choose specific weight matrix set among epochs
        """
        if epoch < subset_net:
            raise ValueError("epoch should be same or larger than subset_net")
        self.som_grid.som(self.som_tr.window_data, epoch, init_rate, init_radius, keep_net = epoch > subset_net)
        if epoch == subset_net:
            self.net = self.som_grid.net
        else:
            self.net = self.som_grid.net_path[subset_net - 1, :, :, :]

    def detect_anomaly(
            self, label = None, threshold = "cltlind",
            level = .9, clt_test = "bh", mfdr = None, power = None,
            bootstrap = 1, clt_map = False, neighbor = None
    ):
        """
        :param label: anomaly and normal label list
        :param threshold: threshold for detection - mean, quantile, radius, kmeans, ztest, clt, cltlind, or anova
        :param level: what quantile to use. Change this for ztest, clt, or cltlind threshold
        :param clt_test: what multiple testing method to use for clt and cltlind - bh, invest, or gai
        :param mfdr: eta of alpha-investing
        :param power: rho of GAI
        :param bootstrap: bootstrap sample number. If 1, bootstrap not performed
        :param clt_map: use mapped codebook for clt and cltlind?
        :param neighbor: radius - use neighboring nodes when clt_map
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if len(label) != 2:
            raise ValueError("label should have 2 elements")
        self.label = label
        thr_types = [
            "quantile", "radius", "mean", "inv_som",
            "kmeans", "hclust",
            "ztest",
            "clt", "cltlind",
            "anova"
        ]
        if threshold not in thr_types:
            raise ValueError("Invalid threshold. Expected one of: %s" % thr_types)
        test_types = ["bh", "invest", "gai"]
        if clt_test not in test_types:
            raise ValueError("Invalid clt_test. Expected on of: %s" % test_types)
        som_anomaly = None
        # threshold with mapping
        if threshold == "quantile" or threshold == "mean" or threshold == "radius":
            anomaly_threshold = None
            dist_anomaly = None
            # normal data
            if self.som_grid.project is None:
                normal_distance = np.asarray(
                    [self.som_grid.dist_weight(self.som_tr.window_data, i) for i in tqdm(range(self.som_tr.window_data.shape[0]), desc = "mapping")]
                )
                self.som_grid.dist_normal = normal_distance[:, 0]
                self.som_grid.project = normal_distance[:, 1]
            # online data
            if self.project is None:
                som_dist_calc = np.asarray(
                    [self.dist_uarray(i) for i in tqdm(range(self.som_te.window_data.shape[0]), desc="mapping online set")]
                )
                dist_anomaly = som_dist_calc[:, 0]
                self.project = som_dist_calc[:, 1]
            # thresholding
            if threshold == "quantile":
                anomaly_threshold = np.quantile(self.som_grid.dist_normal, 2/3)
                som_anomaly = dist_anomaly > anomaly_threshold
            elif threshold == "mean":
                anomaly_threshold = np.mean(self.som_grid.dist_normal)
                som_anomaly = dist_anomaly > anomaly_threshold
            elif threshold == "radius":
                anomaly_threshold = neighbor
                normal_project = np.unique(self.som_grid.project)
                from_normal = self.som_grid.dci[normal_project.astype(int), :]
                anomaly_project = np.full(
                    (normal_project.shape[0], self.net.shape[0]),
                    fill_value = False,
                    dtype = bool
                )
                for i in tqdm(range(normal_project.shape[0]), desc = "neighboring"):
                    anomaly_project[i, :] = from_normal[i, :].flatten() > anomaly_threshold
                anomaly_node = np.argwhere(anomaly_project.sum(axis = 0, dtype = bool))
                som_anomaly = np.isin(self.project, anomaly_node)
        # threshold without mapping
        if threshold == "inv_som":
            som_anomaly = self.inverse_som()
        elif threshold == "kmeans":
            cluster = np.random.choice(np.arange(2), self.som_te.window_data.shape[0])
            cluster_change = cluster + 1
            centroid1 = np.empty((self.net.shape[1], self.net.shape[2]))
            centroid2 = centroid1
            while not np.array_equal(cluster, cluster_change):
                normal_array = np.append(
                    self.net, self.som_te.window_data[cluster == 0, :, :], axis = 0
                )
                anom_array = self.som_te.window_data[cluster == 1, :, :]
                centroid1 = np.mean(normal_array, axis = 0)
                centroid2 = np.mean(anom_array, axis = 0)
                cluster_change = cluster
                for i in range(self.som_te.window_data.shape[0]):
                    dist1 = self.dist_mat(self.som_te.window_data[i, :, :], centroid1)
                    dist2 = self.dist_mat(self.som_te.window_data[i, :, :], centroid2)
                    if dist1 <= dist2:
                        cluster[i] = 0
                    else:
                        cluster[i] = 1
            som_anomaly = np.full(self.som_te.window_data.shape[0], fill_value = False, dtype = bool)
            som_anomaly[cluster == 0] = True
        elif threshold == "hclust":
            som_anomaly = self.hclust_divisive()
        elif threshold == "ztest":
            net_stand = self.net
            # standardize codebook otherwise input standardized
            if not self.standard:
                net_stand = self.net.reshape((-1, self.net.shape[2]))
                scaler = StandardScaler()
                net_stand = scaler.fit_transform(net_stand).reshape(self.net.shape)
            dist_anomaly = np.asarray(
                [self.dist_codebook(net_stand, k) for k in tqdm(range(self.som_te.window_data.shape[0]), desc = "codebook distance")]
            )
            som_anomaly = dist_anomaly > chi2.ppf(level, self.som_te.window_data.shape[1])
        elif threshold == "clt" or threshold == "cltlind":
            if clt_map:
                if self.som_grid.project is None:
                    normal_distance = np.asarray(
                        [self.som_grid.dist_weight(self.som_tr.window_data, i) for i in tqdm(range(self.som_tr.window_data.shape[0]), desc = "mapping")]
                    )
                    self.som_grid.dist_normal = normal_distance[:, 0]
                    self.som_grid.project = normal_distance[:, 1]
                # mapped nodes
                normal_project, proj_count = np.unique(self.som_grid.project, return_counts = True)
                # neighboring nodes
                if neighbor is not None:
                    proj_dist = np.argwhere(
                        self.som_grid.dci[normal_project.astype(int), :] <= neighbor
                    )
                    normal_project, neighbor_count = np.unique(proj_dist[:, 0], return_counts = True)
                    proj_count = np.repeat(proj_count, neighbor_count)
                    normal_project = np.unique(proj_dist[:, 1])
                # corresponding codebook
                net_stand = self.net[normal_project.astype(int), :, :]
            else:
                net_stand = self.net
                proj_count = np.repeat(1, self.net.shape[0])
            if bootstrap == 1:
                normal_distance = np.asarray(
                    [self.dist_normal(net_stand, j) for j in tqdm(range(net_stand.shape[0]), desc = "pseudo-population")]
                )
            else:
                normal_distance = np.asarray(
                    [self.dist_bootstrap(net_stand, j, bootstrap) for j in tqdm(range(net_stand.shape[0]), desc = "pseudo-population")]
                )
            # online set
            dist_anomaly = np.asarray(
                [self.dist_codebook(net_stand, k, w = proj_count) for k in tqdm(range(self.som_te.window_data.shape[0]), desc = "codebook distance")]
            )
            if threshold == "clt":
                # mu = mean(mu1, ..., muN)
                clt_mean = np.average(normal_distance[:, 0], weights = proj_count)
                # sigma = sqrt(sigma1 ** 2 + ... + sigmaN ** 2) / N
                clt_sd = np.sqrt(
                    np.sum(
                        normal_distance[:, 1] * proj_count
                    )
                ) / net_stand.shape[0]
                # sqrt(n) (dbar - mu) -> N(0, sigma2)
                self.dstat = np.sqrt(net_stand.shape[0]) * (dist_anomaly - clt_mean) / clt_sd
            else:
                clt_mean = np.average(normal_distance[:, 0], weights = proj_count)
                sn = np.sqrt(
                    np.sum(normal_distance[:, 1] * proj_count)
                )
                # not iid - lindeberg clt sn = sqrt(sum(sigma2 ** 2)) => sum(xi - mui) / sn -> N(0, 1)
                self.dstat = net_stand.shape[0] * (dist_anomaly - clt_mean) / sn
            pvalue = 1 - norm.cdf(self.dstat)
            if clt_test == "bh":
                alpha = 1 - level
                alpha /= self.som_te.window_data.shape[0]
                # Benjamini–Hochberg - i * alpha / N for ordered p-value
                alpha *= np.arange(self.som_te.window_data.shape[0]) + 1
                # Benjamini–Hochberg 1 find the largest k - p(k) <= alpha(k)
                # 2 reject every H(j), j = 1, ..., k
                p_ordered = np.argsort(pvalue)
                test = np.argwhere(pvalue[p_ordered] <= alpha)
                som_anomaly = np.full(self.som_te.window_data.shape[0], fill_value = False, dtype = bool)
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
                for j in tqdm(range(self.som_te.window_data.shape[0]), desc = "alpha-investing"):
                    h0 = j + 1
                    alphaj = wealth / (1 + h0 - k)
                    rj = pvalue[j] <= alphaj # boolean - 1 if reject, 0 if accept
                    som_anomaly = np.append(som_anomaly, rj)
                    wealth += - (1 - rj) * alphaj / (1 - alphaj) + rj * alpha
                    # wealth += (1 - rj) * np.log(1 - alphaj) + rj * (alpha + np.log(1 - pvalue[j]))
                    k = (1 - rj) * k + rj * h0 # the most recently rejected hypothesis
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
                for j in tqdm(range(self.som_te.window_data.shape[0]), desc = "gai"):
                    # relative200 scheme - phi(k) = w(k - 1) / 10, k = 1, 2, ...
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
        elif threshold == "anova":
            if bootstrap == 1:
                normal_distance = np.asarray(
                    [self.dist_normal(self.net, j) for j in tqdm(range(self.net.shape[0]), desc = "pseudo-population")]
                )
            else:
                normal_distance = np.asarray(
                    [self.dist_bootstrap(self.net, j, bootstrap) for j in tqdm(range(self.net.shape[0]), desc = "pseudo-population")]
                )
            pop_mean = np.average(normal_distance[:, 0])
            # treatment - obs of online set
            trt = self.som_te.window_data.shape[0]
            dij = np.empty((self.net.shape[0], trt))
            for k in tqdm(range(trt), desc = "treatment structure"):
                dij[:, k] = self.dist_dij(self.net, k)
            num = np.sum(dij.shape)
            # treatment mean
            dist_anomaly = np.average(dij, axis = 0)
            # post-anova comparison of means - scheffe
            mse = np.sum(
                np.square(dij - dist_anomaly) / (num - trt)
            )
            std_err = np.sqrt(
                mse / self.net.shape[0]
            )
            # (dbar - sec * sqrt((a-1) * F(a-1, N-a), dbar - sec * sqrt((a-1) * F(a-1, N-a))
            dstat = (dist_anomaly - pop_mean) / std_err
            som_anomaly = dstat >= np.sqrt((trt - 1) * f.ppf(level, trt - 1, num - trt))
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

    def resample_normal(self, boot_num):
        """
        :param boot_num: number of bootsrap samples
        :return: index array for bootstraped samples for normal tensor
        """
        n = self.som_tr.window_data.shape[0]
        id_tr = np.arange(n)
        id_resample = np.random.choice(
            id_tr,
            size = n * boot_num,
            replace = True
        ).reshape((boot_num, n))
        return id_resample

    def dist_bootstrap(self, codebook, node, boot_num):
        """
        :param codebook: transformed codebook matrices
        :param node: node index
        :param boot_num: number of bootsrap samples
        :return: average and sd of distances between normal som matrix and chosen weight matrix
        """
        bootstrap = self.resample_normal(boot_num)
        n = self.som_tr.window_data.shape[0]
        dist_moment = np.empty((boot_num, 2))
        for b in tqdm(range(boot_num), desc = "bootstrap"):
            resample = self.som_tr.window_data[bootstrap[b, :].astype(int), :, :]
            dist_b = np.asarray(
                [self.dist_mat(codebook[node, :, :], resample[i, :, :]) for i in tqdm(range(n), desc = "mean and sd")]
            )
            dist_moment[b, 0] = np.average(dist_b)
            dist_moment[b, 1] = np.var(dist_b)
        return np.average(dist_moment, axis = 0)

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

    def inverse_som(self):
        """
        :return: SOM online data set to codebook matrix -> True if empty grid (anomaly)
        """
        n = self.som_te.window_data.shape[0]
        if np.sqrt(n).is_integer():
            xdim = np.sqrt(n)
            ydim = xdim
        else:
            xdim = int(np.sqrt(n))
            ydim = n / xdim
            while not ydim.is_integer():
                xdim -= 1
                ydim = n / xdim
        online_kohonen = kohonen(
            data = self.net, xdim = int(xdim), ydim = int(ydim), topo = self.topo,
            neighbor = self.h, dist = self.d, decay = self.decay
        )
        online_kohonen.net = self.som_te.window_data
        online_kohonen.som(
            data = self.net, epoch = self.som_grid.epoch,
            init_rate = self.som_grid.initial_learn, init_radius = self.som_grid.initial_r
        )
        if online_kohonen.project is None:
            online_distance = np.asarray(
                [online_kohonen.dist_weight(self.som_te.window_data, i) for i in tqdm(range(self.som_te.window_data.shape[0]), desc="mapping")]
            )
            online_kohonen.project = online_distance[:, 1]
        som_map = np.arange(n)
        return np.isin(som_map, online_kohonen.project, invert = True)

    def hclust_divisive(self):
        """
        :return: divisive hierarchical clustering
        """
        dim1 = self.net.shape[0]
        dim2 = self.som_te.window_data.shape[0]
        # distances between codebook cluster and online observations
        wt_dist = np.empty(dim2)
        for i in tqdm(range(dim2), desc = "codebook vs online"):
            wt_dist[i] = np.average([self.dist_mat(self.net[j, :, :], self.som_te.window_data[i, :, :]) for j in range(dim1)])
        # distance matrix among online observations
        online_pair = np.array(
            np.meshgrid(
                np.arange(dim2),
                np.arange(dim2)
            )
        ).reshape((2, dim2 * dim2)).T
        to_fill = online_pair[online_pair[:, 0] < online_pair[:, 1]]
        data_dist = np.zeros((dim2, dim2))
        for i in tqdm(range(to_fill.shape[0]), desc = "online distance"):
            data_dist[to_fill[i, 0], to_fill[i, 1]] = self.dist_mat(
                self.som_te.window_data[to_fill[i, 0], :, :],
                self.som_te.window_data[to_fill[i, 1], :, :]
            )
        data_dist = data_dist + data_dist.T
        # append first column and row as codebook cluster
        ave_linkage = np.empty((1 + dim2, 1 + dim2))
        ave_linkage[:, range(1, 1 + dim2)] = np.append(wt_dist, data_dist).reshape((1 + dim2, dim2))
        ave_linkage[:, 0] = np.append(0, wt_dist)
        # choose h cluster
        h_cluster = np.argmax(
            ave_linkage.mean(axis=1)
        )
        # send to cluster h until ave_dist from h > ave_dist from g
        h_diff = 1
        while h_diff >= 0:
            if h_cluster.ndim == 0:
                test = ave_linkage[:, h_cluster.astype(int)] - np.delete(ave_linkage, h_cluster, axis = 1).mean(axis = 1)
            else:
                test = ave_linkage[:, h_cluster.astype(int)].mean(axis = 1) - np.delete(ave_linkage, h_cluster, axis = 1).mean(axis = 1)
            h_diff = np.max(test)
            h_cluster = np.append(h_cluster, np.argmax(test))
        h_cluster = h_cluster[:-1]
        g_cluster = np.arange(ave_linkage.shape[0])
        divisive = np.isin(g_cluster, h_cluster)
        if not divisive[0]:
            divisive = np.invert(divisive)
        return np.delete(divisive, 0)

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
        help = "True label dataset file"
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
        default = 50,
        help = "Number of x-grid (Default = 50)"
    )
    parser.add_argument(
        "-y", "--ygrid",
        type = int,
        default = 50,
        help = "Number of y-grid (Default = 50)"
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
    parser.add_argument(
        "--subset",
        type = int,
        help = "Subset codebook matrix set among epochs (Default = epoch number)"
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
        "-b", "--bootstrap",
        type = str,
        default = 1,
        help = "Bootstrap sample numbers (Default = 1, bootstrap not performed)"
    )
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
        help = "Multiple testing method - gai (default), invest, or bh"
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
    output_list = None
    dstat_file = None
    if str(args.output).strip().find(",") != -1:
        output_list = str(args.output).strip().split(",")
        output_file = output_list[0]
        dstat_file = output_list[1]
    else:
        output_file = args.output
    cols = None
    if args.column is not None:
        cols = str(args.column).strip().split(',')
        cols = range(int(cols[0]), int(cols[1]))
    if args.eval is not None:
        print_eval = True
        true_file = args.eval
        target_names = ["anomaly", "normal"]
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
    subset_net = epoch
    init_rate = args.alpha
    init_radius = args.radius
    if args.subset is not None:
        subset_net = args.subset
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
    boot = args.bootstrap
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
    som_anomaly = SomDetect(normal_file, online_file, cols, standard,
                            window_size, jump_size,
                            xdim, ydim, topo, neighbor, dist, decay, seed)
    som_anomaly.learn_normal(epoch = epoch, init_rate = init_rate, init_radius = init_radius, subset_net = subset_net)
    som_anomaly.detect_anomaly(label = label, threshold = threshold,
                               level = ztest_opt, clt_test = multiple_test, mfdr = eta, power = rho,
                               bootstrap = boot, clt_map = proj, neighbor = neighbor_node)
    som_anomaly.label_anomaly()
    anomaly_df = pd.DataFrame({".pred": som_anomaly.anomaly})
    anomaly_df.to_csv(output_file, index = False, header = False)
    if dstat_file is not None:
        dstat_df = pd.DataFrame({".som": som_anomaly.dstat})
        dstat_df.to_csv(dstat_file, index = False, header = False)
        window_df = pd.DataFrame({".pred": som_anomaly.window_anomaly})
        window_df.to_csv(dstat_file.replace(".csv", "_pred.csv"), index = False, header = False)
    print("")
    print("process for %.2f seconds================================================\n" %(time.time() - start_time))
    # files
    print("Files-------------------------------------")
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
    if epoch > subset_net:
        print("Subset weight matrix of: ", subset_net)
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


if __name__ == '__main__':
    np.set_printoptions(precision = 3)
    main()
