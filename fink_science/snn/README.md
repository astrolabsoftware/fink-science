# SuperNNova classifier

This module returns the probability of an alert to be a SNe Ia using [SuperNNova](https://github.com/supernnova/SuperNNova). The module implicitly assumes that the model has been pre-trained, and it is loaded on-the-fly by all executors.

## Pipeline

The pipeline is very simple for the moment:
- extract time, magnitude, and filter information from alerts
- transform magnitude into flux
- Load a pre-trained model
- call SuperNNova to estimate the probability of each alert to be a SNIa.

The condition to run the fit is to have at least 3 valid measurements in a given filter. Note that we give a probability of 0 (i.e. non-Ia) if an alert has not enough measurements in all of its bands.

## Input data for training (v0)

The input data set is ... (TBD)

## Default model

We include a default model with the module in `data/models/<vanilla...>.pt`. This model is based on the v0 description above, and it is loaded if the user does not specify a custom one.

## Typical call

You would use the classifier the following way:

```python
from pyspark.sql import functions as F

from fink_science.snn.processor import snn_ia
from fink_science.utilities import concat_col

# Load alert data in a DataFrame
df = ...

# Where the model is stored [optional]
model_path = ...

# Required alert columns
what = [
  'jd', 'fid', 'magpsf', 'sigmapsf',
  'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

# Use for creating temp name
prefix = 'c'
what_prefix = [prefix + i for i in what]

# Append temp columns with historical + current measurements
for colname in what:
  df = concat_col(df, colname, prefix=prefix)

# Perform the fit + classification (default model)
args = [F.col(i) for i in what_prefix]
df = df.withColumn('snn', snn_ia(*args))
```

## Todo

TBD
