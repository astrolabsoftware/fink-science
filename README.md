[![pypi](https://img.shields.io/pypi/v/fink-science.svg)](https://pypi.python.org/pypi/fink-science)
[![Sentinel](https://github.com/astrolabsoftware/fink-science/workflows/Sentinel/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3ASentinel)
[![PEP8](https://github.com/astrolabsoftware/fink-science/workflows/PEP8/badge.svg)](https://github.com/astrolabsoftware/fink-science/actions?query=workflow%3APEP8)
[![codecov](https://codecov.io/gh/astrolabsoftware/fink-science/branch/master/graph/badge.svg)](https://codecov.io/gh/astrolabsoftware/fink-science)

# Fink Science

This repository contains science modules used to generate added values to alert collected by the [Fink broker](https://github.com/astrolabsoftware/fink-broker). More information at [https://fink-broker.readthedocs.io/en/latest/science/added_values](https://fink-broker.readthedocs.io/en/latest/science/added_values/).

## Installation

Until version 3.13, you can install the package from pypi:


```bash
pip install fink_science --upgrade
```

For later versions, you will need to install from sources directly using a release tag (see [here](https://github.com/astrolabsoftware/fink-science/releases)):

```
# e.g. installing version 4.3.0
export TMPDIR=somewhere_with_disk_space
pip install --no-dependencies --cache-dir=$TMPDIR --build $TMPDIR https://github.com/astrolabsoftware/fink-science/archive/4.3.0.zip
```
