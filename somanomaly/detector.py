import numpy as np
import pandas as pd
import sys
import getopt
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
    3. online SomData - Make online data-st to SomData
    3. Foreach row (0-axis) of online SomData:
        distance from each U-array
        compare with kohonen.sigma (radius)
        if every value is larger than threshold, anomaly
    """

    def __init__(
            self, path_normal, path_online, cols = None, window_size = 60, jump_size = 60,
            xdim = 20, ydim = 20, topo = "rectangular", neighbor = "gaussian", dist = "frobenius", seed = None
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
        :param seed: Random seed
        """
        self.som_tr = SomData(path_normal, cols, window_size, jump_size)
        self.som_te = SomData(path_online, cols, window_size, jump_size)
        self.som_grid = kohonen(self.som_tr.window_data, xdim, ydim, topo, neighbor, dist, seed)
        # anomaly
        self.label = None
        self.window_anomaly = np.empty(self.som_te.window_data.shape[0])
        self.anomaly = np.empty(self.som_te.n)

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
        :param threshold: threshold for detection - quantile, radius or mean
        :return: Anomaly detection
        """
        if label is None:
            label = [True, False]
        if len(label) != 2:
            raise ValueError("label should have 2 elements")
        self.label = label
        dist_anomaly = np.asarray([self.dist_uarray(i) for i in range(self.som_te.window_data.shape[0])])
        thr_types = ["quantile", "radius", "mean"]
        if threshold not in thr_types:
            raise ValueError("Invalid threshold. Expected one of: %s" % thr_types)
        if threshold == "quantile":
            dist_normal = np.asarray([self.dist_normal(i) for i in range(self.som_tr.window_data.shape[0])])
            threshold = np.quantile(dist_normal, 2 / 3)
        elif threshold == "radius":
            threshold = self.som_grid.sigma
        else:
            dist_normal = np.asarray([self.dist_normal(i) for i in range(self.som_tr.window_data.shape[0])])
            threshold = np.mean(dist_normal)
        som_anomaly = dist_anomaly > threshold
        self.window_anomaly[som_anomaly] = self.label[0]
        self.window_anomaly[np.logical_not(som_anomaly)] = self.label[1]

    def dist_uarray(self, index):
        """
        :param index: Row index for online data set
        :return: minimum distance between online data set and weight matrix
        """
        dist_wt = np.asarray([self.som_grid.dist_mat(self.som_te.window_data, index, j) for j in range(self.som_grid.net.shape[0])])
        return np.min(dist_wt)

    def dist_normal(self, index):
        """
        :param index: Row index for normal data set
        :return: every distance between normal som matrix and weight matrix
        """
        return np.asarray([self.som_grid.dist_mat(self.som_tr.window_data, index, j) for j in range(self.som_grid.net.shape[0])])

    def label_anomaly(self):
        win_size = self.som_te.window_data.shape[1]
        jump_size = (self.som_te.n - win_size) // (self.som_te.window_data.shape[0] - 1)
        # first assign by normal
        self.anomaly = np.repeat(self.label[1], self.anomaly.shape[0])
        for i in range(self.window_anomaly.shape[0]):
            if self.window_anomaly[i] == self.label[0]:
                for j in range(i * jump_size, i * jump_size + win_size):
                    if self.anomaly[j] != self.label[0]:
                        self.anomaly[j] = self.label[0]


def main(argv):
    normal_file = ""
    online_file = ""
    output_file = ""
    cols = None
    window_size = 60
    jump_size = 60
    xdim = 20
    ydim = 20
    topo = "rectangular"
    neighbor = "gaussian"
    dist = "frobenius"
    seed = None
    label = [1, 0]
    threshold = "mean"
    epoch = 100
    init_rate = None
    init_radius = None
    try:
        opts, args = getopt.getopt(argv, "hn:o:m:c:w:j:x:y:t:f:d:s:l:p:e:a:r:",
                                   ["help",
                                    "Normal file=", "Online file=", "Output file=", "column index list=(default:None)",
                                    "Window size=(default:60)", "Jump size=(default:60)",
                                    "x-grid=(default:20)", "y-grid=(default:20)", "topology=(default:rectangular)",
                                    "Neighborhood function=(default:gaussian)", "Distance=(default:frobenius)",
                                    "Random seed=(default:None)", "Label=(default:[1,0])", "Threshold=(default:mean)",
                                    "Epoch number=(default:100)",
                                    "Initial learning rate=(default:0.05)", "Initial radius=(default:function)"])
    except getopt.GetoptError:
        print("Getopterror")
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-h" or opt == "--help":
            print("python detector.py -n <normal_file> -o <online_file> -c <column_range> {-w} <window_size> {-j} <jump_size> {-x} <x_grid> {-y} <y_grid> {-t} <topology> {-f} <neighborhood> {-d} <distance> {-l} <label> {-p} <threshold> {-e} <epoch> {-a} <init_rate> {-r} <init_radius>")
            message = """-h or --help: help
            -n: Normal data set file
            -o: Online data set file
            -m: Output file
            -c: first and the last column indices to read, e.g. 1, 5
            Default = None
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
            -s: random seed
            Default = current system time
            -l: anomaly and normal label
            Default = 1,0
            -p: threshold method - quantile, radius, or mean
            Default = mean
            -e: epoch number
            Default = 100
            -a: initial learning ratio
            Default = 0.05
            -r: initial radius of BMU neighborhood
            Default = 2/3 quantile of every distance between nodes
            """
            print(message)
            sys.exit()
        elif opt in ("-n"):
            normal_file = arg
        elif opt in ("-o"):
            online_file = arg
        elif opt in ("-m"):
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
        elif opt in ("-s"):
            seed = int(arg)
        elif opt in ("-l"):
            label = str(arg).strip().split(',')
            label = range(int(label[0]), int(label[1]))
        elif opt in ("-p"):
            threshold = arg
        elif opt in ("-e"):
            epoch = int(arg)
        elif opt in ("-a"):
            init_rate = float(arg)
        elif opt in ("-r"):
            init_radius = float(arg)
    som_anomaly = SomDetect(normal_file, online_file, cols,
                            window_size, jump_size,
                            xdim, ydim, topo, neighbor, dist, seed)
    som_anomaly.learn_normal(epoch = epoch, init_rate = init_rate, init_radius = init_radius)
    som_anomaly.detect_anomaly(label = label, threshold = threshold)
    som_anomaly.label_anomaly()
    anomaly_df = pd.DataFrame(som_anomaly.anomaly)
    anomaly_df.to_csv(output_file, index = False, header = False)


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    main(sys.argv[1:])
