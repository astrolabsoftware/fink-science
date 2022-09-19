# Features for anomaly detection

## Extraction of light curve features

This module adds new column **lc_features** with different features, calculated for each filter, data type of the column is `Map(int filter_id -> Struct(str feature1: double value, etc.))`. Based on [light_curve](https://github.com/light-curve) library. Arxiv link - [paper](https://arxiv.org/pdf/2012.01419.pdf).

Current list of features:

```txt
A1.0.2  Mean, A2.0.12  Mean
A1.0.3  Weighted Mean, A2.0.29  Weighted mean
A1.0.4  Standard Deviation, A2.0.27  Standard Deviation
A1.0.7  Median

A2.0.1  Amplitude
A2.0.2  Beyond𝑛Std
A2.0.3  Cusum
A2.0.6  Inter-percentile range
A2.0.7  Kurtosis
A2.0.8  Linear Trend
A2.0.9  Linear Fit
A2.0.10  Magnitude Percentage Ratio for (0.4, 0.05) and (0.2, 0.1) - 'default' values

A2.0.11  Maximum Slope
A2.0.13  Median Absolute Deviation
A2.0.14  Median Buffer Range Percentage
A2.0.15  Percent Amplitude
Mean Variance
Anderson Darling Normal
A2.0.25  Reduced 𝜒2
A2.0.26  Skew
A2.0.28  Stetson𝐾
```
