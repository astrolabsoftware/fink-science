# Random Forest Classifier

This module returns the probability of an alert to be a SNe Ia using a Random Forest Classifier (binary classification). It benefits from the work in [Ishida et al](https://arxiv.org/abs/1804.03765). The module implicitly assumes that the model has been pre-trained, and it is loaded on-the-fly by all executors (the model is currently ~20 MB).

## Pipeline

The pipeline is very simple for the moment:
- extract time, magnitude, and filter information from alerts
- transform magnitude into flux
- for each alert, and for each filter, perform a Bazin fit to estimate the parameters `[A, B, t0, tfall, trise]`
- Load a pre-trained model (Random Forest), and for each alert estimate the probability to be a SNIa.


## Model (v0)

The input data set is SNPCC (DES) that has been _alertified_. It contains 21,319 light-curves with Ia, Ib, Ic, and II. The light-curves typically span several months, with 4 observation bands (g, r, i, z). To make it more realistic in our runs, we degraded the light-curve to look like ZTF alert data: 30 days, and 3 observation bands (g, r, i).

The distribution of the model is currently done via CVMFS (i.e. it is pre-loaded on CVMFS, and each executor can see it).
