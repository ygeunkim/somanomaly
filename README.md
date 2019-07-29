# SomAnomaly

**Online SOM Detector** - Anomaly detection using Self-Organizing Maps

## Motivation

![Process time series](docs/som_data.png)

Given multivariate time series, we are trying to outlying pattern. This represents anomaly.

1. Slide window
2. Bind the windows

Then we get 3d tensor. Now fit Self-organizing maps to this form of data-set. Different with ordinary SOM structure, we use input **matrices**, not vectors.

## Anomaly detection

Build 3d array for online data-set. Compute each distance between codebook matrix and window matrix.
If it is larger than threshold, the window is detected as anomaly.

## Building

```
git clone https://github.com/ygeunkim/onlinesom.git
cd onlinesom
python setup.py build
python setup.py install
```

### Usage

In terminal, you can run *Online SOM detector* using `onlinesom/detector.py`:

```
cd onlinesom
python detector.py -n <normal_file> -o <online_file> -c <column_range> {-w} <window_size> {-j} <jump_size> {-x} <x_grid> {-y} <y_grid> {-t} <topology> {-f} <neighborhood> {-d} <distance> {-l} <label> {-p} <threshold> {-e} <epoch> {-a} <init_rate> {-r} <init_radius>
```

### File path

#### Input file

```
-n Normal dataset file
-o Online dataset file
-c Column index to read - start,end (Default = every column)
```

#### Output file

```
-m Anomaly detection output file
```

### Training SOM

```
-w Window size (Default = 60)
-j Shift size (Default = 60)
-x Number of x-grid (Default = 20)
-y Number of y-grid (Default = 20)
-t Topology of SOM output space - rectangular (default) or hexagonal
-f Neighborhood function - gaussian (default) or bubble
-d Distance function - frobenius (default) or nuclear
-e Epoch number (Default = 100)
```

### Detecting

```
-l Anomaly and normal labels, e.g. 1,0 (default)
-p Threshold method - mean (default), 0.75 quantile, or radius
```
