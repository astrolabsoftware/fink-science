[![pypi](https://img.shields.io/pypi/v/fink-science.svg)](https://pypi.python.org/pypi/fink-science) [![Build Status](https://travis-ci.org/astrolabsoftware/fink-science.svg?branch=master)](https://travis-ci.org/astrolabsoftware/fink-science) [![codecov](https://codecov.io/gh/astrolabsoftware/fink-science/branch/master/graph/badge.svg)](https://codecov.io/gh/astrolabsoftware/fink-science)

# Fink Science

This repository contains science modules used to generate added values to alert collected by the [Fink broker](https://github.com/astrolabsoftware/fink-broker). It currently contains:

- `xmatch`: returns the SIMBAD closest counterpart of an alert, based on position.
- `random_forest_snia`: returns the probability of an alert to be a SNe Ia using a Random Forest Classifier (binary classification)

## How to contribute

Learn how to [design](https://fink-broker.readthedocs.io/en/latest/tutorials/create-science-module/) your science module, and integrate it inside the Fink broker.

## Installation

If you want to install the package (broker deployment), you can just pip it:

```
pip install fink_science
```
