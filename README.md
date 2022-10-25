[![pypi](https://img.shields.io/pypi/v/fink-science.svg)](https://pypi.python.org/pypi/fink-science)
[![Sentinel](https://github.com/astrolabsoftware/fink-science/workflows/Sentinel/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3ASentinel)
[![PEP8](https://github.com/astrolabsoftware/fink-science/workflows/PEP8/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3APEP8)
[![codecov](https://codecov.io/gh/astrolabsoftware/fink-science/branch/master/graph/badge.svg)](https://codecov.io/gh/astrolabsoftware/fink-science)

# Fink Science

This repository contains science modules used to generate added values to alert collected by the [Fink broker](https://github.com/astrolabsoftware/fink-broker).

## ZTF alert stream

It currently contains:

| Source | Field in Fink alerts | Type | Contents |
|:---|:-----|:-------|:--------|
| [fink_science/xmatch](fink_science/xmatch)|`cdsxmatch`++ | string | Counterpart (cross-match) from any CDS catalog or database using the [CDS xmatch service](http://cdsxmatch.u-strasbg.fr/xmatch). Contains also crossmatch to the [General Catalog of Variable Stars](http://www.sai.msu.su/groups/cluster/gcvs/gcvs/) and the [International Variable Star Index](https://www.aavso.org/vsx/), [3HSP](https://www.ssdc.asi.it/3hsp/), [4LAC DR3](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4LACDR3/). |
| [fink_science/random_forest_snia](fink_science/random_forest_snia)| `rf_snia_vs_nonia` | float | Probability to be a rising SNe Ia based on Random Forest classifier (1 is SN Ia). Based on https://arxiv.org/abs/2111.11438 |
| [fink_science/snn](fink_science/snn)| `snn_snia_vs_nonia` | float | Probability to be a SNe Ia based on [SuperNNova](https://supernnova.readthedocs.io/en/latest/) classifier (1 is SN Ia). Based on https://arxiv.org/abs/1901.06384 |
| [fink_science/snn](fink_science/snn)| `snn_sn_vs_all` | float | Probability to be a SNe based on [SuperNNova](https://supernnova.readthedocs.io/en/latest/) classifier (1 is SNe). Based on https://arxiv.org/abs/1901.06384 |
| [fink_science/microlensing](fink_science/microlensing)| `mulens`| float | Probability score to be a microlensing event by [LIA](https://github.com/dgodinez77/LIA) |
| [fink_science/asteroids](fink_science/asteroids)| `roid` | int | Determine if the alert is a Solar System object |
| [fink_science/kilonova](fink_science/kilonova)| `rf_kn_vs_nonkn` | float | probability of an alert to be a kilonova using a Random Forest Classifier (binary classification). |
| [fink_science/nalerthist](fink_science/nalerthist)| `nalerthist` | int | Number of detections contained in each alert (current+history). Upper limits are not taken into account. |
| [fink_science/ad_features](fink_science/ad_features)| `lc_*` | dict[int, array<double>] | Numerous [light curve features](https://arxiv.org/pdf/2012.01419.pdf#section.A1) used in astrophysics. |
| [fink_science/agn](fink_science/agn)| `rf_agn_vs_nonagn` | float | Probability to be an AGN based on Random Forest classifier (1 is AGN). |

You will find README in each subfolder describing the module.

## ELASTiCC stream (Rubin-like simulated data)

These modules are being tested for Rubin era on the LSST-DESC ELASTiCC data challenge:

| Source | Field in Fink alerts | Type | Contents |
|:---|:-----|:-------|:--------|
| [fink_science/agn_elasticc](fink_science/agn_elasticc)| `rf_agn_vs_nonagn` | float | Probability to be an AGN based on Random Forest classifier (1 is AGN). |
| [fink_science/random_forest_snia](fink_science/random_forest_snia)| `rf_snia_vs_nonia` | float | Probability to be a rising SNe Ia based on Random Forest classifier (1 is SN Ia). Based on https://arxiv.org/abs/2111.11438 |
| [fink_science/snn](fink_science/snn)| `snn_snia_vs_nonia` | float | Probability to be a SNe Ia based on [SuperNNova](https://supernnova.readthedocs.io/en/latest/) classifier (1 is SN Ia). Based on https://arxiv.org/abs/1901.06384 |
| [fink_science/snn](fink_science/snn)| `broad` | array[float] | Broad classifier based on SNN. Returns [class, max(prob)]. |
| [fink_science/cats](fink_science/cats)| `fine` | array[float] | Fine classifier based on the CBPF Algorithm for Transient Search. Returns [class, max(prob)]. |

## Under development

| Source | Field in Fink alerts | Type | Contents |
|:---|:-----|:-------|:--------|
| [fink_science/t2](fink_science/t2)| `t2` | array[float] | Classifier based on Transformers. Based on https://arxiv.org/abs/2105.06178 |

## How to contribute

Learn how to [design](tutorial/) your science module, and integrate it inside the Fink broker.

## Installation

If you want to install the package (broker deployment), you can just pip it:

```
pip install fink_science
```
