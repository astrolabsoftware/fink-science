[![Sentinel](https://github.com/astrolabsoftware/fink-science/actions/workflows/run_test.yml/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions/workflows/run_test.yml)
[![PEP8](https://github.com/astrolabsoftware/fink-science/workflows/PEP8/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3APEP8)
[![codecov](https://codecov.io/gh/astrolabsoftware/fink-science/branch/master/graph/badge.svg)](https://codecov.io/gh/astrolabsoftware/fink-science)

# Fink Science

This repository contains science modules used to generate added values to alert collected by the [Fink broker](https://github.com/astrolabsoftware/fink-broker). More information at:

- ZTF: [https://doc.ztf.fink-broker.org/en/latest/broker/science_modules/](https://doc.ztf.fink-broker.org/en/latest/broker/science_modules/).
- LSST: [https://doc.lsst.fink-broker.org/science/science_modules/](https://doc.lsst.fink-broker.org/science/science_modules/)

## Installation

`fink-science` can be easily installed from sources directly using a release tag (see [here](https://github.com/astrolabsoftware/fink-science/releases)):

```
# e.g. installing version 8.41.0
export TMPDIR=somewhere_with_disk_space
pip install git+https://github.com/astrolabsoftware/fink-science@8.41.0
```
