# Copyright 2022-2025 Fink Software
# Author: Etienne Russeil
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
from line_profiler import profile

from fink_science.rubin.slsn.classifier import slsn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
from fink_science.tester import spark_unit_tests

from fink_science import __file__


@pandas_udf(DoubleType())
@profile
def slsn_rubin(diaObjectId, cmidpointMjdTai, cpsfFlux, cpsfFluxErr, cband, ra, dec):
    """High level spark wrapper for the slsn classifier on Rubin data

    Parameters
    ----------
    diaObjectId: Spark DataFrame Column
        Identification numbers of the objects
    cmidPoinTai: Spark DataFrame Column
        JD times (vectors of floats)
    cpsfFlux, cpsfFluxErr: Spark DataFrame Columns
        Flux and flux error from photometry (vectors of floats)
    cband: Spark DataFrame Column
        Filter IDs (vectors of str)
    ra: Spark DataFrame Column
        Right ascension of the objects
    dec: Spark DataFrame Column
        Declination of the objects

    Returns
    -------
    np.array
        ordered probabilities of being a slsn
        Return 0 if the minimum points number is not respected.

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
    >>> args = ["diaObject.diaObjectId"]
    >>> args += [F.col(i) for i in what_prefix]
    >>> args += ["diaSource.ra", "diaSource.dec"]
    >>> df = df.withColumn('preds', slsn_rubin(*args))
    >>> df.filter(df['preds'] == 0).count()
    50
    """
    data = pd.DataFrame({
        "diaObjectId": diaObjectId,
        "cmidpointMjdTai": cmidpointMjdTai,
        "cpsfFlux": cpsfFlux,
        "cpsfFluxErr": cpsfFluxErr,
        "cband": cband,
        "ra": ra,
        "dec": dec,
    })

    proba = slsn_classifier(data, False)
    return pd.Series(proba)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    rubin_alert_sample = "file://{}/data/alerts/or4_lsst7.1".format(path)
    globs["rubin_alert_sample"] = rubin_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
