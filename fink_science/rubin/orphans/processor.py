# Copyright 2019-2025 AstroLab Software
# Authors: Marina Masson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pandas as pd
import numpy as np

from fink_science.rubin.orphans.classifier import orphan_classifier
from fink_science.rubin.orphans.basic_functions import flux_to_mag

from fink_science import __file__
from fink_science.tester import spark_unit_tests

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType


@pandas_udf(FloatType())
def orphan_grb(cmidpointMjdTai, cpsfFlux, cpsfFluxErr, cband):
    """Filter to extract orphan GRB candidates

    Notes
    -----
    The filter is based on different features calculated from the light curve:
    - Duration between the first detection and the peak
    - Increase and decrease rates
    - Colour between different filter pairs
    - 4 parameters and chi2 from fitting the light curve

    A Boosting Decision Tree classifier was trained to discriminate
    orphans using these features

    Parameters
    ----------
    cmidpointMjdTai: pandas.Series of list of float
        Concatenated time in MJD for alerts
    cpsfFlux: pandas.Series of list of float, or Nan
        Concatenated fluxes for alerts
    cpsfFluxErr: pandas.Series of list of float, or Nan
        Concatenated errors on the fluxes for alerts
    cband: pandas.Series of list of int
        Concatenated filters for alerts

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the probability
        of each alert to be an orphan

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> df = spark.read.format('parquet').load(rubin_alert_sample)

    # Required alert columns
    >>> what = ['midpointMjdTai', 'psfFlux', 'psfFluxErr', 'band']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     df = concat_col(
    ...         df, colname, prefix=prefix,
    ...         current='diaSource', history='prvDiaForcedSources')

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> df = df.withColumn("prob", orphan_grb(*args))

    # No match obviously from the current test set
    >>> df.filter(df["prob"] > 0.0).count()
    0
    """
    # Compute diff mag from diff flux
    cabmags = cpsfFlux.apply(lambda x: flux_to_mag(x))
    cabmagserr = (
        cpsfFluxErr.apply(lambda x: np.array(x))
        / cpsfFlux.apply(lambda x: np.array(x))
        * 1.0857
    )

    valid = cabmags.apply(lambda x: True if len(~np.isnan(x)) >= 5 else False)
    if np.sum(valid) == 0:
        return pd.Series(np.zeros(len(cmidpointMjdTai), dtype=float))
    proba_orphan = orphan_classifier(cmidpointMjdTai, cabmags, cabmagserr, cband, valid)

    return pd.Series(proba_orphan)


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()
    path = os.path.dirname(__file__)

    rubin_alert_sample = "file://{}/data/alerts/or4_lsst7.1".format(path)
    globs["rubin_alert_sample"] = rubin_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
