# Quiescent state of blazar detection

This module adds a new column containing quantities that indicate if the last observation of a given blazar can classify it as a **quiescent state**.
The available blazar set comes from the list of blazars that are planned to be monitored by the CTAO collaboration from the start of the South and North sites.

These quantities are:
* The sliding mean of standardized flux of the last but new alert over a specific threshold computed for the source beforehand
* The sliding mean of standardized flux of the new alert over the same threshold computed for the source beforehand
* The standardized flux of the last alert over the same threshold

If the last two ratio are below one, the source is considered to be in a quiescent state.
If the source is in a quiescent state but the first ratio is above one, this quiescent state is considered to have just started.
By default, all value are set to -1 as it is an unphysical value.
