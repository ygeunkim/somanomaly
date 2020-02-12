# <img alt="SomAnomaly" src="docs/somanomaly_icon.png" height="60">

Anomaly detection using [Self-Organizing Maps](https://en.wikipedia.org/wiki/Self-organizing_map)

## Building

This module requires the following.

- Numpy: [www.numpy.org](https://www.numpy.org)
- pandas: [pandas.pydata.org](https://pandas.pydata.org)
- scipy: [www.scipy.org](https://www.scipy.org)
- scikit-learn: [scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- plotly: [plot.ly/python/](https://plot.ly/python/)
- matplotlib: [matplotlib.org](https://matplotlib.org)
- tqdm: [tqdm.github.io](https://tqdm.github.io)
- argparse: [github.com/ThomasWaldmann/argparse](https://github.com/ThomasWaldmann/argparse/)

```
cd somanomaly
python setup.py build
python setup.py install
```

### Usage

In command line, you can run *SomAnomaly* using `somanomaly/detector.py`:

```
cd somanomaly
python detector.py [-h] [-c COLUMN] [-e EVAL] [--log] [--standardize]
                   [-w WINDOW] [-j JUMP] [-x XGRID] [-y YGRID] [-p PROTOTYPE]
                   [-n NEIGHBORHOOD] [-m METRIC] [-d DECAY] [-s SEED]
                   [-i ITER] [-a ALPHA] [-r RADIUS] [-l LABEL] [-u THRESHOLD]
                   [-o] [-q MULTIPLE] [-1] [-2] [-3]
                   normal online output
```

The following is a description of each argument.

### Positional arguments

```
-h, --help  show the help message and exit
```

#### Input file

For now, this function reads only `*.csv` files using `pandas.read_csv()`

```
normal  Training dataset file
online  Test dataset file
```

Warning: *this function requires exactly same form of both files.
If you use `-c` option, it will be applied to both normal data set file and online data-set file.*

In case of `-c`, follow the python `range(start, end)` function.
Then the columns from `start + 1` to `end` in the file will be read.

```
-c  Column index to read - start,end (Default = every column)
```

#### Output file

```
output  Anomaly detection output file
```

This file does not have any column header or row index. You can output multiple files using comma.

- anomaly detection
- SomAnomaly statistic
- anomaly detection for each window

If you add one more file using comma, two more files will be generated.

### Optional arguments

#### True value

```
-e, --eval  True label file
```

If this file is provided, evaluation result (precision, recall, and F1-score) will be printed.

#### SOM

```
--standardize  Standardize both data sets
-w, --window  Window size (Default = 30)
-j, --jump  Shift size (Default = 30)
-x, --xgrid  Number of x-grid (Default = 50)
-y, --ygrid  Number of y-grid (Default = 50)
-p, --prototype  Topology of SOM output space - hexagonal (default) or rectangular
-n, --neighborhood  Neighborhood function - gaussian (default), triangular, or bubble
-m, --metric  Distance function - frobenius (default), nuclear, mahalanobis, or eros
-d, --decay  Decaying function - exponential (default) or linear
-s, --seed  Random seed (Default = system time)
-i, --iter  Epoch number (Default = 50)
-a, --alpha  Initial learning rate (Default = 0.1)
-r, --radius  Initial radius of BMU neighborhood (Default = 2/3 quantile of every distance between nodes)
```

#### Detecting anomaly

For anomaly detection, we use test based on Gaussian distribution.

```
-l, --label  Anomaly and normal labels, e.g. 1,0 (default)
-u, --threshold  Significant level for the test
```

Multiple testing:

```
-o, --overfit  Use only mapped codebook if specified
-q, --multiple  Multiple testing method - gai (default), invest, bon, or bh
``` 

Both `invest` and `gai` have option for the detector. See each

- *Foster, D. P., & Stine, R. A. (2008). α‐investing: a procedure for sequential control of expected false discoveries. Journal of the Royal Statistical Society Series B-Statistical Methodology, 70(2), 429–444. http://doi.org/10.1111/j.1467-9868.2007.00643.x*
- *Aharoni, E., & Rosset, S. (2014). Generalized α‐investing: definitions, optimality results and application to public databases. Journal of the Royal Statistical Society Series B-Statistical Methodology, 76(4), 771–794. http://doi.org/10.1111/rssb.12048*

`invest,number` or `gai,number` will control an *eta* in mFDR. Additionally, `gai+number` will control the upper bound of power for `gai` (Generalized alpha-investing).

#### Plot

You can see the following plots if writing each parameter.

```
-1, --eror  Plot reconstruction error for each epoch
-2, --heat  Plot heatmap of SOM
-3, --pred  Plot heatmap of projection onto normal SOM
```
