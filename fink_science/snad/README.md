# SNAD

## Extraction of light curve features

This module adds many new columns with different features for data analysis, ML, etc. Based on [light_curve](https://github.com/light-curve) library. Arxiv link - [paper](https://arxiv.org/pdf/2012.01419.pdf).

Current list of columns:

```txt
lc_amplitude,
lc_beyond_1_std,
lc_linear_fit_slope,
lc_linear_fit_slope_sigma,
lc_linear_fit_reduced_chi2,
lc_mean,lc_median,
lc_standard_deviation,
lc_cusum,
lc_excess_variance,
lc_mean_variance,
lc_kurtosis,
lc_maximum_slope,
lc_skew,
lc_eta,
lc_anderson_darling_normal,
lc_chi2,
lc_inter_percentile_range_10,
lc_median_buffer_range_percentage_10,
lc_percent_difference_magnitude_percentile_10,
lc_median_absolute_deviation,lc_percent_amplitude,
lc_eta_e,lc_linear_trend,
lc_linear_trend_sigma,
lc_linear_trend_noise,
lc_stetson_K,
lc_weighted_mean,
lc_magnitude_percentage_ratio_40_5,
lc_magnitude_percentage_ratio_20_10
```
