# Quiescent state of blazar detection

This module adds a new column containing quantities that indicate whether the last observation of a given blazar can be classified as an 'extreme state'.
The blazar set available comes from the list of blazars planned for monitoring by the CTAO collaboration at the start of operations at the South and North sites.

These quantities are:
* the sliding mean of the standardised flux of the last alert over a specific low state threshold, computed for the source in advance.
* the standardised flux of the last alert over the same low state threshold;
* the sliding mean of the standardised flux of the last alert over a specific high state threshold, computed for the source beforehand.
* the standardised flux of the last alert over the same high state threshold;
* the quantile corresponding to the last measurement on the cumulative density function (CDF).

If the first two ratios are below one, the source is considered to be in a quiescent state. Conversely, if the last two ratios are above one, the source is considered to be in a flaring state. The corresponding CDF position is only computed if the blazar is considered to be in an extreme state.
By default, all values are set to -1, as this is an unphysical value.
