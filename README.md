# Online SOM Detector

Anomaly detection using *Self-Organizing Maps*

## Motivation

![Process time series](docs/som_data.png)

Given multivariate time series, we are trying to outlying pattern. This represents anomaly.

1. Slide window
2. Bind the windows

Then we get 3d tensor. Now fit Self-organizing maps to this form of data-set. Different with ordinary SOM structure, we use input **matrices**, not vectors.

## Anomaly detection

Build 3d array for online data-set. Compute each distance between codebook matrix and window matrix.
If it is larger than threshold, the window is detected as anomaly.
