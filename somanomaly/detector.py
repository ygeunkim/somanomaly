import numpy as np
import pandas as pd
import sys
import getopt
import plotly.graph_objs as go
from tqdm import tqdm
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
            self, path_normal, path_online, cols = None, window_size = 60, jump_size = 60,
            xdim = 20, ydim = 20, topo = "rectangular", neighbor = "gaussian",
            dist = "frobenius", decay = "exponential", seed = None
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
        :param decay: decaying learning rate and radius - exponential or linear
        :param seed: Random seed
        """
        self.som_tr = SomData(path_normal, cols, window_size, jump_size)
        self.som_te = SomData(path_online, cols, window_size, jump_size)
        self.som_grid = kohonen(self.som_tr.window_data, xdim, ydim, topo, neighbor, dist, decay, seed)
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
        self.project = np.empty(self.som_te.window_data.shape[0])

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
        som_dist_calc = np.asarray([self.dist_uarray(i) for i in tqdm(range(self.som_te.window_data.shape[0]), desc = "mapping online set")])
        dist_anomaly = som_dist_calc[:, 0]
        self.project = som_dist_calc[:, 1]
        thr_types = ["quantile", "radius", "mean", "inv_som"]
        som_anomaly = None
        if threshold not in thr_types:
            raise ValueError("Invalid threshold. Expected one of: %s" % thr_types)
        anomaly_threshold = None
        if threshold == "quantile":
            # dist_normal = np.asarray([self.dist_normal(i) for i in range(self.som_tr.window_data.shape[0])])
            # threshold = np.quantile(dist_normal, 2 / 3)
            anomaly_threshold = np.quantile(self.som_grid.dist_normal, 2/3)
        elif threshold == "mean":
            # dist_normal = np.asarray([self.dist_normal(i) for i in range(self.som_tr.window_data.shape[0])])
            # threshold = np.mean(dist_normal)
            anomaly_threshold = np.mean(self.som_grid.dist_normal)
        elif threshold == "radius":
            anomaly_threshold = self.som_grid.initial_r
            normal_project = np.unique(self.som_grid.project)
            from_normal = self.som_grid.dci[normal_project.astype(int), :]
            anomaly_project = np.full((normal_project.shape[0], self.som_grid.net.shape[0]), fill_value = False, dtype = bool)
            for i in range(normal_project.shape[0]):
                anomaly_project[i, :] = from_normal[i, :].flatten() > anomaly_threshold
            anomaly_node = np.argwhere(anomaly_project.sum(axis = 0, dtype = bool))
            som_anomaly = np.isin(self.project, anomaly_node)
        elif threshold == "inv_som":
            som_anomaly = self.inverse_som()
        if som_anomaly is None:
            som_anomaly = dist_anomaly > anomaly_threshold
        self.window_anomaly[som_anomaly] = self.label[0]
        self.window_anomaly[np.logical_not(som_anomaly)] = self.label[1]

    def dist_uarray(self, index):
        """
        :param index: Row index for online data set
        :return: minimum distance between online data set and weight matrix
        """
        # normal_map = np.unique(self.som_grid.project)
        dist_wt = np.asarray([self.som_grid.dist_mat(self.som_te.window_data, index, j) for j in tqdm(range(self.som_grid.net.shape[0]), desc = "bmu")])
        return np.min(dist_wt), np.argmin(dist_wt)

    def dist_normal(self, index):
        """
        :param index: Row index for normal data set
        :return: every distance between normal som matrix and weight matrix
        """
        return np.asarray([self.som_grid.dist_mat(self.som_tr.window_data, index, j) for j in range(self.som_grid.net.shape[0])])

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
        som_map = np.arange(n)
        return np.isin(som_map, online_kohonen.project, invert = True)

    def label_anomaly(self):
        win_size = self.som_te.window_data.shape[1]
        jump_size = (self.som_te.n - win_size) // (self.som_te.window_data.shape[0] - 1)
        # first assign by normal
        self.anomaly = np.repeat(self.label[1], self.anomaly.shape[0])
        for i in tqdm(range(self.window_anomaly.shape[0]), desc = "anomaly unit change"):
            if self.window_anomaly[i] == self.label[0]:
                for j in tqdm(range(i * jump_size, i * jump_size + win_size), desc = "observation unit"):
                    if self.anomaly[j] != self.label[0]:
                        self.anomaly[j] = self.label[0]

    def plot_heatmap(self):
        """
        :return: heatmap of projection onto normal SOM
        """
        xdim = self.som_grid.net_dim[0]
        ydim = self.som_grid.net_dim[1]
        neuron_grid = np.empty((xdim, ydim))
        node_id = 0
        for j in range(ydim):
            for i in range(xdim):
                neuron_grid[i, j] = (self.project == node_id).sum()
                node_id += 1
        fig = go.Figure(
            data = go.Heatmap(z = neuron_grid, colorscale = "Viridis")
        )
        fig.show()


def main(argv):
    normal_file = ""
    online_file = ""
    output_file = ""
    cols = None
    # training arguments
    window_size = 60
    jump_size = 60
    xdim = 20
    ydim = 20
    topo = "rectangular"
    neighbor = "gaussian"
    dist = "frobenius"
    decay = "exponential"
    seed = None
    epoch = 100
    init_rate = None
    init_radius = None
    # detection arguments
    label = [1, 0]
    threshold = "mean"
    # plot options
    print_error = False
    print_heat = False
    print_projection = False
    try:
        opts, args = getopt.getopt(argv, "hn:o:p:c:w:j:x:y:t:f:d:g:s:l:m:e:a:r:123",
                                   ["help",
                                    "Normal file=", "Online file=", "Output file=", "column index list=(default:None)",
                                    "Window size=(default:60)", "Jump size=(default:60)",
                                    "x-grid=(default:20)", "y-grid=(default:20)", "topology=(default:rectangular)",
                                    "Neighborhood function=(default:gaussian)", "Distance=(default:frobenius)",
                                    "Decay=(default:exponential)",
                                    "Random seed=(default:None)", "Label=(default:[1,0])", "Threshold=(default:mean)",
                                    "Epoch number=(default:100)",
                                    "Initial learning rate=(default:0.5)", "Initial radius=(default:function)",
                                    "Plot reconstruction error",
                                    "Plot heatmap for SOM",
                                    "Plot heatmap of projection onto normal SOM"])
    except getopt.GetoptError as err:
        print(err)
        usage_message = """python detector.py -n <normal_file> -o <online_file> {-c} <column_range> -p <output_file>
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
            message = """-h or --help: help
            -n: Normal data set file
            -o: Online data set file
            -c: first and the last column indices to read, e.g. 1, 5
                Default = None
            -p: Output file
            -w: window size
                Default = 60
            -j: shift size
                Default = 60
            -x: number of x-grid
                Default = 20
            -y: number of y-grid
                Default = 20
            -t: topology of SOM output space - rectangular or hexagonal
                Default = rectangular
            -f: neighborhood function - gaussian or bubble
                Default = gaussian
            -d: distance function - frobenius or nuclear
                Default = frobenius
            -g: decaying function - exponential or linear
                Default = exponential
            -s: random seed
                Default = current system time
            -e: epoch number
                Default = 100
            -a: initial learning ratio
                Default = 0.5
            -r: initial radius of BMU neighborhood
                Default = 2/3 quantile of every distance between nodes
            -l: anomaly and normal label
                Default = 1,0
            -m: threshold method - quantile, radius, mean, or inv_som
                Default = mean
            -1: plot reconstruction error path
                Default = do not plot
            -2: plot heatmap of SOM
                Default = do not plot
            -3: plot heatmap of projection onto normal SOM
                Default = do not plot
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
    som_anomaly = SomDetect(normal_file, online_file, cols,
                            window_size, jump_size,
                            xdim, ydim, topo, neighbor, dist, decay, seed)
    som_anomaly.learn_normal(epoch = epoch, init_rate = init_rate, init_radius = init_radius)
    som_anomaly.detect_anomaly(label = label, threshold = threshold)
    som_anomaly.label_anomaly()
    anomaly_df = pd.DataFrame(som_anomaly.anomaly)
    anomaly_df.to_csv(output_file, index = False, header = False)
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
