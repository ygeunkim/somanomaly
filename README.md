# SomAnomaly

**Online SOM Detector** - Anomaly detection using Self-Organizing Maps

## Building

```
git clone https://github.com/ygeunkim/somanomaly.git
cd somanomaly
python setup.py build
python setup.py install
```

### Usage

In terminal, you can run *Online SOM detector* using `somanomaly/detector.py`:

```
cd somanomaly
python detector.py -n <normal_file> -o <online_file> -c <column_range> 
                        {-w} <window_size> {-j} <jump_size> {-x} <x_grid> {-y} <y_grid> 
                        {-t} <topology> {-f} <neighborhood> {-d} <distance> {-l} <label> 
                        {-p} <threshold> {-e} <epoch> {-a} <init_rate> {-r} <init_radius>
```

The following is a description of each argument.

```
-h, --help  show the help message
```

### File path

#### Input file

For now, this function reads only `*.csv` files.

```
-n  Normal dataset file
-o  Online dataset file
-c  Column index to read - start,end (Default = every column)
```

Warning: *this function requires exactly same form of both files.
If you use `-c` option, it will be applied to both normal data set file and online data-set file.*

In case of `-c`, follow the python `range(start, end)` function.
Then the columns from `start + 1` to `end` in the file will be read.

#### Output file

```
-m  Anomaly detection output file
```

This file does not have any column header or row index.

### SOM

Options for training and detection have default values, respectively.
So all these are optional arguments.

#### Training SOM

```
-w  Window size (Default = 60)
-j  Shift size (Default = 60)
-x  Number of x-grid (Default = 20)
-y  Number of y-grid (Default = 20)
-t  Topology of SOM output space - rectangular (default) or hexagonal
-f  Neighborhood function - gaussian (default) or bubble
-d  Distance function - frobenius (default) or nuclear
-s  Random seed (Default = system time)
-e  Epoch number (Default = 100)
-a  Initial learning rate (Default = 0.05)
-r  Initial radius of BMU neighborhood (Default = 2/3 quantile of every distance between nodes)
```

#### Detecting anomaly

```
-l  Anomaly and normal labels, e.g. 1,0 (default)
-p  Threshold method - mean (default), 0.75 quantile, or radius
```

***

## Motivation

### Pre-processing

<p align="center">
    <img width="70%" height="43.26%" src="docs/som_data.png" >
</p>

Given multivariate time series, we are trying to find outlying pattern. This represents anomaly.

1. Slide window
2. Bind the windows

Then we get 3d tensor. Now fit Self-organizing maps to this form of data-set. Different with ordinary SOM structure, we use input **matrices**, not vectors.

The algorithm requires computing *distance between matrices* - input matrix and weight (codebook) matrix.

<p align="center">
    <img width="70%" height="43.26%" src="docs/distance.png">
</p>

### Anomaly detection

1. Build 3d array for online data-set.
2. Compute each distance between weight (codebook) matrix.
3. If the minimum is larger than threshold, the window is detected as anomaly.

<p align="center">
    <img width="70%" height="43.26%" src="docs/som_detect.png">
</p>
