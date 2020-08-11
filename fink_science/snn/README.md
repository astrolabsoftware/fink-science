# SuperNNova classifier

This module returns the probability of an alert to be a SNe Ia using [SuperNNova](https://github.com/supernnova/SuperNNova). The module implicitly assumes that the model has been pre-trained, and it is loaded on-the-fly by all executors.

## Pipeline

The pipeline is very simple for the moment:
- extract time, magnitude, and filter information from alerts
- transform magnitude into flux
- Load a pre-trained model
- call SuperNNova to estimate the probability of each alert to be a SNIa.

## Input data for training (v0)

The input data set is ... (TBD)

## Default model

We include two default models with the module:
- fink_science/data/models/snn_models/snn_snia_vs_nonia/model.pt
- fink_science/data/models/snn_models/snn_sn_vs_all/model.pt


These models are based on the v0 description above, and `snn_snia_vs_nonia` is loaded if the user does not specify a custom one.

## Typical call

You would use the classifier the following way:

```python
from fink_science.utilities import concat_col
from pyspark.sql import functions as F

df = ...

# Required alert columns
what = ['jd', 'fid', 'magpsf', 'sigmapsf']

# Use for creating temp name
prefix = 'c'
what_prefix = [prefix + i for i in what]

# Append temp columns with historical + current measurements
for colname in what:
  df = concat_col(df, colname, prefix=prefix)

# Perform the fit + classification (default model)
args = ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf', F.lit('snn_snia_vs_nonia')]
df = df.withColumn('pIa', snn_ia(*args))

# Note that we can also specify the path to a model (needs to be distributed though)
args = [F.col(i) for i in ['candid', 'cjd', 'cfid', 'cmagpsf', 'csigmapsf', F.lit('')]] + [F.lit(model_path)]
df = df.withColumn('pIa', snn_ia(*args))
```

## Todo

TBD
