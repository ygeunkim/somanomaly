import argparse
import time
import numpy as np
import pandas as pd
import re
from somanomaly.detector import SomDetect
from sklearn.metrics import classification_report

def load_data(path):
    if re.search(r'\.csv$', path, re.IGNORECASE):
        return pd.read_csv(path, header = None)
    elif re.search(r'\.parquet$', path, re.IGNORECASE):
        return pd.read_parquet(path)
    elif re.search(r'\.feather$', path, re.IGNORECASE):
        return pd.read_feather(path)
    elif re.search(r'\.xlsx$', path, re.IGNORECASE):
        return pd.read_excel(path, header = None)
    elif re.search(r'\.json$', path, re.IGNORECASE):
        return pd.read_json(path)
    # elif re.search(r'\.pkl$', file_path, re.IGNORECASE):
    #     with open(file_path, 'rb') as file:
    #         return pickle.load(file)
    else:
        raise ValueError("Unsupported file format")

def save_data(df, path):
    if re.search(r'\.csv$', path, re.IGNORECASE):
        df.to_csv(path, index = False, header = False)
    elif re.search(r'\.parquet$', path, re.IGNORECASE):
        df.to_parquet(path, index = False)
    elif re.search(r'\.feather$', path, re.IGNORECASE):
        df.reset_index(drop = True).to_feather(path)
    elif re.search(r'\.xlsx$', path, re.IGNORECASE):
        df.to_excel(path, index = False, header = False)
    elif re.search(r'\.json$', path, re.IGNORECASE):
        df.to_json(path, orient = 'records')
    else:
        raise ValueError("Unsupported file format")

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
    parser.add_argument(
        "--log",
        help = "Log transform",
        action = "store_true"
    )
    parser.add_argument(
        "--logstat",
        help = "Log2 stat",
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
        help = "Distance function - frobenius (default), spectral, nuclear, mahalanobis, or eros"
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
    # parser.add_argument(
    #     "--subset",
    #     type = int,
    #     help = "Subset codebook matrix set among epochs (Default = epoch number)"
    # )
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
    stat_log = args.logstat
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
    # subset_net = epoch
    init_rate = args.alpha
    init_radius = args.radius
    # if args.subset is not None:
    #     subset_net = args.subset
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
    if normal_list is None:
        som_anomaly = SomDetect(normal_file, online_file, cols, standard,
                                window_size, jump_size, test_log,
                                xdim, ydim, topo, neighbor, dist, decay, seed)
        som_anomaly.learn_normal(epoch = epoch, init_rate = init_rate, init_radius = init_radius)
        som_anomaly.detect_anomaly(label = label, threshold = threshold,
                                   level = ztest_opt, clt_test = multiple_test, mfdr = eta, power = rho, log_stat = stat_log,
                                   bootstrap = boot, clt_map = proj, neighbor = neighbor_node)
        som_anomaly.label_anomaly()
        anomaly_df = pd.DataFrame({".pred": som_anomaly.anomaly})
        # anomaly_df.to_csv(output_file, index = False, header = False)
        save_data(anomaly_df, output_file)
        if dstat_file is not None:
            dstat_df = pd.DataFrame({".som": som_anomaly.dstat})
            # dstat_df.to_csv(dstat_file, index = False, header = False)
            save_data(dstat_df, dstat_file)
            window_df = pd.DataFrame({".pred": som_anomaly.window_anomaly})
            # window_df.to_csv(dstat_file.replace(".csv", "_pred.csv"), index = False, header = False)
            save_data(window_df, re.sub(r'(\.\w+)$', r'_pred\1', dstat_file))
    else:
        anomaly_pred = SomDetect.detect_block(
            normal_list, online_list, col_list,
            standard, window_size, jump_size, test_log,
            xdim, ydim, topo, neighbor, dist, decay, seed,
            epoch, init_rate, init_radius, label, ztest_opt, multiple_test, eta, rho, stat_log
        )
        anomaly_df = pd.DataFrame({".pred": anomaly_pred})
        # anomaly_df.to_csv(output_file, index = False, header = False)
        save_data(anomaly_df, output_file)
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
    # if epoch > subset_net:
    #     print("Subset weight matrix of: ", subset_net)
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
            true_anomaly = load_data(true_file)
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
            true_anomaly = load_data(true_file)
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