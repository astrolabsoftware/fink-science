# Features for anomaly detection

## Extraction of light curve features

This module adds new column **lc_features** with different features, calculated for each filter, data type of the column is `Map(int filter_id -> Struct(str feature1: double value, etc.))`. Based on [light_curve](https://github.com/light-curve) library. Arxiv link - [paper](https://arxiv.org/pdf/2012.01419.pdf).

Current list of features:

```txt
 1 - mean  
 2 - weighted_mean  
 3 - standard_deviation  
 4 - median  
 5 - amplitude  
 6 - beyond_1_std  
 7 - cusum  
 8 - inter_percentile_range_10  
 9 - kurtosis  
 10 - linear_trend  
 11 - linear_trend_sigma  
 12 - linear_trend_noise  
 13 - linear_fit_slope  
 14 - linear_fit_slope_sigma  
 15 - linear_fit_reduced_chi2  
 16 - magnitude_percentage_ratio_40_5  
 17 - magnitude_percentage_ratio_20_10  
 18 - maximum_slope  
 19 - median_absolute_deviation  
 20 - median_buffer_range_percentage_10  
 21 - percent_amplitude  
 22 - mean_variance  
 23 - anderson_darling_normal  
 24 - chi2  
 25 - skew  
 26 - stetson_K
```
