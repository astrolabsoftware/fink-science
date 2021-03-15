[![pypi](https://img.shields.io/pypi/v/fink-science.svg)](https://pypi.python.org/pypi/fink-science)
[![Sentinel](https://github.com/astrolabsoftware/fink-science/workflows/Sentinel/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3ASentinel)
[![PEP8](https://github.com/astrolabsoftware/fink-science/workflows/PEP8/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3APEP8)
[![codecov](https://codecov.io/gh/astrolabsoftware/fink-science/branch/master/graph/badge.svg)](https://codecov.io/gh/astrolabsoftware/fink-science)

# Fink Science

This repository contains science modules used to generate added values to alert collected by the [Fink broker](https://github.com/astrolabsoftware/fink-broker). It currently contains:

- `xmatch`: returns the SIMBAD closest counterpart of an alert, based on position.
- `random_forest_snia`: returns the probability of an alert to be a SNe Ia using a Random Forest Classifier (binary classification)
- `snn`: returns the probability of an alert to be a SNe Ia using [SuperNNova](https://github.com/supernnova/SuperNNova). Two pre-trained models:
  - `snn_snia_vs_nonia`: Ia vs core-collapse SNe
  - `snn_sn_vs_all`: SNe vs. anything else (variable stars and other categories in the training)
- `microlensing`: returns the predicted class (among microlensing, variable star, cataclysmic event, and constant event) & probability of an alert to be a microlensing event in each band using [LIA](https://github.com/dgodinez77/LIA).
- `asteroids`: Determine if the alert is a Solar System Object (experimental).
- `nalerthist`: Number of detections contained in each alert (current+history). Upper limits are not taken into account.
- `kilonova`: returns the probability of an alert to be a kilonova using a Random Forest Classifier (binary classification).

You will find README in each subfolder describing the module.

## How to contribute

Learn how to [design](https://fink-broker.readthedocs.io/en/latest/tutorials/create-science-module/) your science module, and integrate it inside the Fink broker.

## Installation

If you want to install the package (broker deployment), you can just pip it:

```
pip install fink_science
```
