import numpy as np
import pandas as pd
import sys
import getopt
import plotly.graph_objs as go
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
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
        # self.project = np.empty(self.som_te.window_data.shape[0])
        self.project = None

    def learn_normal(self, epoch = 100, init_rate = None, init_radius = None):
        """
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        """
        self.som_grid.som(self.som_tr.window_data, epoch, init_rate, init_radius)

    def detect_anomaly(self, label = None, threshold = "quantile"):
        """
        :param label: anomaly and normal label list
        :param threshold: threshold for detection - quantile, radius, mean, or inv_som
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if len(label) != 2:
            raise ValueError("label should have 2 elements")
        self.label = label
        thr_types = ["quantile", "radius", "mean", "inv_som", "kmeans", "hclust", "ztest"]
        if threshold not in thr_types:
            raise ValueError("Invalid threshold. Expected one of: %s" % thr_types)
        som_anomaly = None
        # threshold with mapping
        if threshold == "quantile" or threshold == "mean" or threshold == "radius":
            anomaly_threshold = None
            dist_anomaly = None
            # normal data
            if self.som_grid.project is None:
                normal_distance = np.asarray(
                    [self.som_grid.dist_weight(self.som_tr.window_data, i) for i in tqdm(range(self.som_tr.window_data.shape[0]), desc="mapping")]
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
                anomaly_threshold = self.som_grid.initial_r
                normal_project = np.unique(self.som_grid.project)
                from_normal = self.som_grid.dci[normal_project.astype(int), :]
                anomaly_project = np.full(
                    (normal_project.shape[0], self.som_grid.net.shape[0]),
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
            centroid1 = np.empty((self.som_grid.net.shape[1], self.som_grid.net.shape[2]))
            centroid2 = centroid1
            while not np.array_equal(cluster, cluster_change):
                normal_array = np.append(
                    self.som_grid.net, self.som_te.window_data[cluster == 0, :, :], axis = 0
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
            net_stand = self.som_grid.net
            # standardize codebook otherwise input standardized
            if not self.standard:
                net_stand = self.som_grid.net.reshape((-1, self.som_grid.net.shape[2]))
                scaler = StandardScaler()
                net_stand = scaler.fit_transform(net_stand).reshape(self.som_grid.net.shape)
            dist_anomaly = np.asarray(
                [self.dist_codebook(net_stand, k) for k in tqdm(range(self.som_te.window_data.shape[0]), desc = "codebook distance")]
            )
            som_anomaly = dist_anomaly > chi2.ppf(.9, self.som_te.window_data.shape[1])
        # label
        self.window_anomaly[som_anomaly] = self.label[0]
        self.window_anomaly[np.logical_not(som_anomaly)] = self.label[1]

    def dist_uarray(self, index):
        """
        :param index: Row index for online data set
        :return: minimum distance between online data set and weight matrix
        """
        dist_wt = np.asarray([self.som_grid.dist_mat(self.som_te.window_data, index, j) for j in tqdm(range(self.som_grid.net.shape[0]), desc = "bmu")])
        return np.min(dist_wt), np.argmin(dist_wt)

    def dist_codebook(self, codebook, index):
        """
        :param codebook: transformed codebook matrices
        :param index: Row index for online data set
        :return: average distance between online data set and weight matrix
        """
        net_num = codebook.shape[0]
        dist_wt = np.asarray(
            [self.dist_mat(self.som_te.window_data[index, :, :], codebook[i, :, :]) for i in tqdm(range(net_num), desc = "averaging")]
        )
        return np.average(dist_wt)

    def dist_normal(self, index):
        """
        :param index: Row index for normal data set
        :return: every distance between normal som matrix and weight matrix
        """
        return np.asarray([self.som_grid.dist_mat(self.som_tr.window_data, index, j) for j in range(self.som_grid.net.shape[0])])

    def dist_mat(self, mat1, mat2):
        """
        :param mat1: Matrix
        :param mat2: Matrix
        :return: distance between mat1 and mat2
        """
        if self.som_grid.dist_func == "frobenius":
            return np.linalg.norm(mat1 - mat2, "fro")
        elif self.som_grid.dist_func == "nuclear":
            return np.linalg.norm(mat1 - mat2, "nuc")
        elif self.som_grid.dist_func == "mahalanobis":
            x = mat1 - mat2
            covmat = np.cov(x, rowvar = False)
            # ss = x.dot(np.linalg.inv(covmat)).dot(x.T)
            w, v = np.linalg.eigh(covmat)
            w[w == 0] += .0001
            covinv = v.dot(np.diag(1 / w)).dot(v.T)
            ss = x.dot(covinv).dot(x.T)
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
            data = self.som_grid.net, xdim = int(xdim), ydim = int(ydim), topo = self.topo,
            neighbor = self.h, dist = self.d, decay = self.decay
        )
        online_kohonen.net = self.som_te.window_data
        online_kohonen.som(
            data = self.som_grid.net, epoch = self.som_grid.epoch,
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
        dim1 = self.som_grid.net.shape[0]
        dim2 = self.som_te.window_data.shape[0]
        # distances between codebook cluster and online observations
        wt_dist = np.empty(dim2)
        for i in tqdm(range(dim2), desc = "codebook vs online"):
            wt_dist[i] = np.average([self.dist_mat(self.som_grid.net[j, :, :], self.som_te.window_data[i, :, :]) for j in range(dim1)])
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
        win_size = self.som_te.window_data.shape[1]
        jump_size = (self.som_te.n - win_size) // (self.som_te.window_data.shape[0] - 1)
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
        fig = go.Figure(
            go.Histogram2d(
                x = x,
                y = y,
                colorscale="Viridis"
            )
        )
        fig.show()


def main(argv):
    normal_file = ""
    online_file = ""
    output_file = ""
    cols = None
    # training arguments
    standard = False
    window_size = 30
    jump_size = 30
    xdim = 50
    ydim = 50
    topo = "rectangular"
    neighbor = "gaussian"
    dist = "frobenius"
    decay = "exponential"
    seed = None
    epoch = 50
    init_rate = None
    init_radius = None
    # detection arguments
    label = [1, 0]
    threshold = "ztest"
    # print_eval
    print_eval = False
    target_names = ["anomaly", "normal"]
    true_file = None
    # plot options
    print_error = False
    print_heat = False
    print_projection = False
    try:
        opts, args = getopt.getopt(argv, "hn:o:p:c:z:iw:j:x:y:t:f:d:g:s:l:m:e:a:r:123",
                                   ["help",
                                    "Normal file=", "Online file=", "Output file=", "column index list=(default:None)",
                                    "True label file",
                                    "Standardize",
                                    "Window size=(default:30)", "Jump size=(default:30)",
                                    "x-grid=(default:50)", "y-grid=(default:50)", "topology=(default:rectangular)",
                                    "Neighborhood function=(default:gaussian)", "Distance=(default:frobenius)",
                                    "Decay=(default:exponential)",
                                    "Random seed=(default:None)", "Label=(default:[1,0])", "Threshold=(default:ztest)",
                                    "Epoch number=(default:50)",
                                    "Initial learning rate=(default:0.5)", "Initial radius=(default:function)",
                                    "Plot reconstruction error",
                                    "Plot heatmap for SOM",
                                    "Plot heatmap of projection onto normal SOM"])
    except getopt.GetoptError as err:
        print(err)
        usage_message = """python detector.py -n <normal_file> -o <online_file> {-c} <column_range>
                                                    -p <output_file> {-z} <true_file>
                                                    {-w} <window_size> {-j} <jump_size> {-x} <x_grid> {-y} <y_grid> 
                                                    {-t} <topology> {-f} <neighborhood> {-d} <distance> {-g} <decay> 
                                                    {-s} <seed> {-e} <epoch> {-a} <init_rate> {-r} <init_radius>
                                                    {-l} <label> {-m} <threshold>
                                                    {-1} {-2} {-3}
        """
        print(usage_message)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h" or opt == "--help":
            message = """Arguments:
            -h or --help: help
File path:
            -n: Normal data set file
            -o: Online data set file
            -c: first and the last column indices to read, e.g. 1,5 --> usecols=range(1,5)
                Default = None (every column)
            -p: Output file
            -z: True label file (optional - if provided, print evaluation)
Training SOM (option):
            -i: standardize data if specified
            -w: window size
                Default = 30
            -j: shift size
                Default = 30
            -x: number of x-grid
                Default = 50
            -y: number of y-grid
                Default = 50
            -t: topology of SOM output space - rectangular or hexagonal
                Default = rectangular
            -f: neighborhood function - gaussian or bubble
                Default = gaussian
            -d: distance function - frobenius, nuclear, or mahalanobis
                Default = frobenius
            -g: decaying function - exponential or linear
                Default = exponential
            -s: random seed
                Default = current system time
            -e: epoch number
                Default = 50
            -a: initial learning ratio
                Default = 0.5
            -r: initial radius of BMU neighborhood
                Default = 2/3 quantile of every distance between nodes
Detecting anomalies (option):
            -l: anomaly and normal label
                Default = 1,0
            -m: threshold method - quantile, radius, mean, inv_som, kmeans, hclust, or ztest
                Default = ztest
Plot if specified:
            -1: plot reconstruction error path
            -2: plot heatmap of SOM
            -3: plot heatmap of projection onto normal SOM
            """
            print(message)
            sys.exit()
        elif opt in ("-n"):
            normal_file = arg
        elif opt in ("-o"):
            online_file = arg
        elif opt in ("-p"):
            output_file = arg
        elif opt in ("-c"):
            cols = str(arg).strip().split(',')
            cols = range(int(cols[0]), int(cols[1]))
        elif opt in ("-z"):
            print_eval = True
            true_file = arg
        elif opt in ("-i"):
            standard = True
        elif opt in ("-w"):
            window_size = int(arg)
        elif opt in ("-j"):
            jump_size = int(arg)
        elif opt in ("-x"):
            xdim = int(arg)
        elif opt in ("-y"):
            ydim = int(arg)
        elif opt in ("-t"):
            topo = arg
        elif opt in ("-f"):
            neighbor = arg
        elif opt in ("-d"):
            dist = arg
        elif opt in ("-g"):
            decay = arg
        elif opt in ("-s"):
            seed = int(arg)
        elif opt in ("-l"):
            label = str(arg).strip().split(',')
            label = range(int(label[0]), int(label[1]))
        elif opt in ("-m"):
            threshold = arg
        elif opt in ("-e"):
            epoch = int(arg)
        elif opt in ("-a"):
            init_rate = float(arg)
        elif opt in ("-r"):
            init_radius = float(arg)
        elif opt in ("-1"):
            print_error = True
        elif opt in ("-2"):
            print_heat = True
        elif opt in ("-3"):
            print_projection = True
    som_anomaly = SomDetect(normal_file, online_file, cols, standard,
                            window_size, jump_size,
                            xdim, ydim, topo, neighbor, dist, decay, seed)
    som_anomaly.learn_normal(epoch = epoch, init_rate = init_rate, init_radius = init_radius)
    som_anomaly.detect_anomaly(label = label, threshold = threshold)
    som_anomaly.label_anomaly()
    anomaly_df = pd.DataFrame({".pred": som_anomaly.anomaly})
    anomaly_df.to_csv(output_file, index = False, header = False)
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
    if print_error:
        som_anomaly.som_grid.plot_error()
    if print_heat:
        som_anomaly.som_grid.plot_heatmap(som_anomaly.som_tr.window_data)
    if print_projection:
        som_anomaly.plot_heatmap()


if __name__ == '__main__':
    np.set_printoptions(precision = 3)
    main(sys.argv[1:])
