import numpy as np
import pandas as pd
import sys
import getopt
import time
from scipy.stats import chi2
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from sklearn.metrics import classification_report
from somanomaly import kohonen
from somanomaly.window import SomData


class KohonenBlock:
    """
    When CPS has blocked structure, you might follow it.
    For each block, fit SOM (with same parameter set) and get the final weight matrix.
    Aggregate them for each node.
    You can get same dimension of matrix as online data-set.
    Detect anomaly in the online data-set as detector.py
    """

    def __init__(
            self, normal_list, online_list, col_list,
            standard, window_size = 60, jump_size = 60,
            xdim = 20, ydim = 20, topo = "rectangular", neighbor = "gaussian",
            dist = "frobenius", decay = "exponential", seed = None,
            epoch = 100, init_rate = None, init_radius = None
    ):
        """
        :param normal_list: List of normal data-set files
        :param online_list: List of online data-set files
        :param col_list: Each column index
        :param standard: standardize both data sets
        :param window_size: window size
        :param jump_size: shift size
        :param xdim: Number of x-grid
        :param ydim: Number of y-grid
        :param topo: Topology of output space
        :param neighbor: Neighborhood function
        :param dist: Distance function
        :param decay: decaying learning rate and radius
        :param seed: Random seed
        :param epoch: epoch number
        :param init_rate: initial learning rate
        :param init_radius: initial radius of BMU neighborhood
        """
        num_tr = len(normal_list)
        num_te = len(online_list)
        self.standard = standard
        self.net_dim = np.array([xdim, ydim])
        # Distance function
        dist_type = ["frobenius", "nuclear", "mahalanobis", "eros"]
        if dist not in dist_type:
            raise ValueError("Invalid dist. Expected one of: %s" % dist_type)
        self.dist_func = dist
        # each block of normal data-set
        som_tr = SomData(normal_list[0], range(col_list[0], col_list[1]), window_size, jump_size)
        if self.standard:
            som_tr.window_data = KohonenBlock.standardize_array(som_tr)
        som_grid = kohonen(som_tr.window_data, xdim, ydim, topo, neighbor, dist, decay, seed)
        self.net = som_grid.net
        j = 2
        for i in tqdm(range(1, num_tr), desc = "aggregate codebook"):
            som_tr = SomData(
                normal_list[i],
                range(col_list[j], col_list[j + 1]),
                window_size, jump_size
            )
            if self.standard:
                som_tr.window_data = KohonenBlock.standardize_array(som_tr)
            som_grid = kohonen(som_tr.window_data, xdim, ydim, topo, neighbor, dist, decay, seed)
            # train SOM in i-th block
            som_grid.som(som_tr.window_data, epoch, init_rate, init_radius, False)
            self.net = np.append(self.net, som_grid.net, axis = 2)
            j += 2
        # Online data-set
        self.som_te = SomData(online_list[0], range(col_list[0], col_list[1]), window_size, jump_size)
        j = 2
        for i in tqdm(range(1, num_te), desc = "online array"):
            self.som_te.window_data = np.append(
                self.som_te.window_data,
                SomData(
                    online_list[i],
                    range(col_list[j], col_list[j + 1]),
                    window_size,
                    jump_size
                ).window_data,
                axis = 2
            )
            j += 2
        if self.standard:
            self.som_te.window_data = KohonenBlock.standardize_array(self.som_te)
        # anomaly
        self.label = None
        self.window_anomaly = np.empty(self.som_te.window_data.shape[0])
        self.anomaly = np.empty(self.som_te.n)

    @staticmethod
    def standardize_array(som_data):
        """
        :param som_data: SomData object
        :return: window_data of standardized SomData
        """
        scaler = StandardScaler()
        tmp_data = som_data.window_data.reshape((-1, som_data.window_data.shape[2]))
        tmp_data = scaler.fit_transform(tmp_data).reshape(som_data.window_data.shape)
        return tmp_data

    def detect_anomaly(self, label = None, threshold = "ztest", chi_opt = .9):
        """
        :param label: anomaly and normal label list
        :param threshold: threshold for detection
        :param chi_opt: what number chi-squared quantile to use. Change this only when ztest threshold
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if len(label) != 2:
            raise ValueError("label should have 2 elements")
        self.label = label
        thr_types = ["kmeans", "ztest"]
        if threshold not in thr_types:
            raise ValueError("Invalid threshold. Expected one of: %s" % thr_types)
        som_anomaly = None
        # detect
        if threshold == "ztest":
            net_stand = self.net
            # standardize codebook otherwise input standardized
            if not self.standard:
                net_stand = self.net.reshape((-1, self.net.shape[2]))
                scaler = StandardScaler()
                net_stand = scaler.fit_transform(net_stand).reshape(self.net.shape)
            dist_anomaly = np.asarray(
                [self.dist_codebook(net_stand, k) for k in
                 tqdm(range(self.som_te.window_data.shape[0]), desc="codebook distance")]
            )
            som_anomaly = dist_anomaly > chi2.ppf(chi_opt, self.som_te.window_data.shape[1])
        elif threshold == "kmeans":
            cluster = np.random.choice(np.arange(2), self.som_te.window_data.shape[0])
            cluster_change = cluster + 1
            centroid1 = np.empty((self.net.shape[1], self.net.shape[2]))
            centroid2 = centroid1
            while not np.array_equal(cluster, cluster_change):
                normal_array = np.append(
                    self.net, self.som_te.window_data[cluster == 0, :, :], axis=0
                )
                anom_array = self.som_te.window_data[cluster == 1, :, :]
                centroid1 = np.mean(normal_array, axis=0)
                centroid2 = np.mean(anom_array, axis=0)
                cluster_change = cluster
                for i in range(self.som_te.window_data.shape[0]):
                    dist1 = self.dist_mat(self.som_te.window_data[i, :, :], centroid1)
                    dist2 = self.dist_mat(self.som_te.window_data[i, :, :], centroid2)
                    if dist1 <= dist2:
                        cluster[i] = 0
                    else:
                        cluster[i] = 1
            som_anomaly = np.full(self.som_te.window_data.shape[0], fill_value=False, dtype=bool)
            som_anomaly[cluster == 0] = True
        # label
        self.window_anomaly[som_anomaly] = self.label[0]
        self.window_anomaly[np.logical_not(som_anomaly)] = self.label[1]

    def dist_mat(self, mat1, mat2):
        """
        :param mat1: Matrix
        :param mat2: Matrix
        :return: distance between mat1 and mat2
        """
        if self.dist_func == "frobenius":
            return np.linalg.norm(mat1 - mat2, "fro")
        elif self.dist_func == "nuclear":
            return np.linalg.norm(mat1 - mat2, "nuc")
        elif self.dist_func == "mahalanobis":
            x = mat1 - mat2
            covmat = np.cov(x, rowvar = False)
            w, v = np.linalg.eigh(covmat)
            w[w == 0] += .0001
            covinv = v.dot(np.diag(1 / w)).dot(v.T)
            ss = x.dot(covinv).dot(x.T)
            return np.sqrt(np.trace(ss))
        elif self.dist_func == "eros":
            x = mat1 - mat2
            covmat = np.cov(x, rowvar = False)
            u, s, vh = randomized_svd(covmat, n_components = covmat.shape[1], n_iter = 1, random_state = None)
            w = s / s.sum()
            ss = np.multiply(vh, w).dot(vh.T)
            return np.sqrt(np.trace(ss))

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

    def label_anomaly(self):
        win_size = self.som_te.window_data.shape[1]
        jump_size = (self.som_te.n - win_size) // (self.som_te.window_data.shape[0] - 1)
        # first assign by normal
        self.anomaly = np.repeat(self.label[1], self.anomaly.shape[0])
        for i in tqdm(range(self.window_anomaly.shape[0]), desc = "anomaly unit change"):
            if self.window_anomaly[i] == self.label[0]:
                self.anomaly[(i * jump_size):(i * jump_size + win_size)] = self.label[0]


def main(argv):
    normal_list = None
    online_list = None
    output_file = ""
    col_list = None
    # training arguments
    standard = False
    window_size = 30
    jump_size = 30
    xdim = 25
    ydim = 25
    topo = "hexagonal"
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
    ztest_opt = .9
    threshold_list = None
    # print_eval
    print_eval = False
    target_names = ["anomaly", "normal"]
    true_file = None
    try:
        opts, args = getopt.getopt(argv, "hn:o:p:c:z:iw:j:x:y:t:f:d:g:s:l:m:e:a:r:k:123",
                                   ["help",
                                    "Normal file=", "Online file=", "Output file=", "column index list=(default:None)",
                                    "True label file",
                                    "Standardize",
                                    "Window size=(default:30)", "Jump size=(default:30)",
                                    "x-grid=(default:25)", "y-grid=(default:25)",
                                    "topology=(default:hexagonal)",
                                    "Neighborhood function=(default:gaussian)", "Distance=(default:frobenius)",
                                    "Decay=(default:exponential)",
                                    "Random seed=(default:None)", "Label=(default:[1,0])", "Threshold=(default:ztest)",
                                    "Epoch number=(default:50)",
                                    "Initial learning rate=(default:0.5)", "Initial radius=(default:function)",
                                    "Subset weight matrix among epochs=(default:50)",
                                    "Plot reconstruction error",
                                    "Plot heatmap for SOM",
                                    "Plot heatmap of projection onto normal SOM"])
    except getopt.GetoptError as err:
        print(err)
        usage_message = """python detector.py -n <normal_file> -o <online_file> {-c} <column_range>
                                                    -p <output_file> {-z} <true_file>
                                                    {-i} {-w} <window_size> {-j} <jump_size> {-x} <x_grid> {-y} <y_grid> 
                                                    {-t} <topology> {-f} <neighborhood> {-d} <distance> {-g} <decay> 
                                                    {-s} <seed> {-e} <epoch> {-a} <init_rate> {-r} <init_radius>
                                                    {-k} <subset_net>
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
            -n: Normal data set file list
            -o: Online data set file list
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
                Default = 25
            -y: number of y-grid
                Default = 25
            -t: topology of SOM output space - rectangular or hexagonal
                Default = hexagonal
            -f: neighborhood function - gaussian or bubble
                Default = gaussian
            -d: distance function - frobenius, nuclear, mahalanobis, or eros
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
            -m: threshold method - kmeans, ztest
                Default = ztest
                Note: if you give use ztest with comma and quantile number such as ztest,0.9, you can change the quantile.
            """
            print(message)
            sys.exit()
        elif opt in ("-n"):
            normal_list = str(arg).strip().split(",")
        elif opt in ("-o"):
            online_list = str(arg).strip().split(",")
        elif opt in ("-p"):
            output_file = arg
        elif opt in ("-c"):
            col_list = list(map(int, str(arg).strip().split(",")))
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
            label = str(arg).strip().split(",")
            label = [int(label[0]), int(label[1])]
        elif opt in ("-m"):
            if str(arg).strip().find(",") != -1:
                threshold_list = str(arg).strip().split(",")
                threshold = threshold_list[0]
                ztest_opt = float(threshold_list[1])
            else:
                threshold = arg
        elif opt in ("-e"):
            epoch = int(arg)
        elif opt in ("-a"):
            init_rate = float(arg)
        elif opt in ("-r"):
            init_radius = float(arg)
    start_time = time.time()
    som_block = KohonenBlock(
        normal_list, online_list, col_list,
        standard, window_size, jump_size,
        xdim, ydim, topo, neighbor, dist, decay, seed,
        epoch, init_rate, init_radius
    )
    som_block.detect_anomaly(label = label, threshold = threshold, chi_opt = ztest_opt)
    som_block.label_anomaly()
    anomaly_df = pd.DataFrame({".pred": som_block.anomaly})
    anomaly_df.to_csv(output_file, index = False, header = False)
    print("")
    print("process for %.2f seconds================================================\n" %(time.time() - start_time))
    # print parameter
    print("SOM parameters----------------------------")
    if standard:
        print("Standardized!")
    print("[Window, jump]: ", [window_size, jump_size])
    print("SOM grid: ", som_block.net_dim)
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
    print("==========================================")
    # evaluation
    if print_eval:
        true_anomaly = pd.read_csv(true_file, header = None)
        true_anomaly = pd.DataFrame.to_numpy(true_anomaly)
        print(
            classification_report(
                true_anomaly, som_block.anomaly,
                labels = label, target_names = target_names
            )
        )


if __name__ == '__main__':
    np.set_printoptions(precision = 3)
    main(sys.argv[1:])
