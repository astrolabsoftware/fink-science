[![Rubin](https://github.com/astrolabsoftware/fink-science/actions/workflows/run_test_rubin.yml/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions/workflows/run_test_rubin.yml)
[![ZTF](https://github.com/astrolabsoftware/fink-science/actions/workflows/run_test_ztf.yml/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions/workflows/run_test_ztf.yml)
[![PEP8](https://github.com/astrolabsoftware/fink-science/workflows/PEP8/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3APEP8)
[![codecov](https://codecov.io/gh/astrolabsoftware/fink-science/branch/master/graph/badge.svg)](https://codecov.io/gh/astrolabsoftware/fink-science)

# Fink Science

This repository contains science modules used to generate added values to alert collected by the [Fink broker](https://github.com/astrolabsoftware/fink-broker). More information on the [documentation](https://fink-broker.readthedocs.io/en/latest/broker/science_modules/).

## Installation

`fink-science` can be easily installed from sources directly using a release tag (see [here](https://github.com/astrolabsoftware/fink-science/releases)):

```
# e.g. installing version 5.7.2
export TMPDIR=somewhere_with_disk_space
pip install --no-dependencies --cache-dir=$TMPDIR https://github.com/astrolabsoftware/fink-science/archive/5.7.2.zip
```
