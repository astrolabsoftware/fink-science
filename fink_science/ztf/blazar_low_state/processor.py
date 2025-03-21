# Copyright 2025 AstroLab Software
# Author: Julian Hamo
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
from line_profiler import profile

import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import MapType, StringType, FloatType
from fink_science.ztf.blazar_low_state.utils import quiescent_state_, BLAZAR_COLS

from fink_science.tester import spark_unit_tests
from fink_science import __file__
import os

RELEASE = 22


@pandas_udf(MapType(StringType(), FloatType()))
@profile
def quiescent_state(
    candid: pd.Series, objectId: pd.Series, cstd_flux: pd.Series, cjd: pd.Series
) -> pd.Series:
    """Returns an array containing blazar features

    Notes
    -----
    Features are:
    m0: The mean over threshold ratio of the last but one alert
    m1: The mean over threshold ratio of the last alert
    m2: The standardized flux over threshold ratio of the last alert

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf, cmagnr,
        csigmagnr, cisdiffpos, cfid, cjd, cstd_flux, csigma_std_flux
    CTAO_blazar: pd.core.frame.DataFrame
        Pandas DataFrame of the monitored sources containing:
        3FGL Name, ZTF Name, Arrays of Medians, Computed Threshold,
        Observed Threshold, Redshift, Final Threshold

    Returns
    -------
    out: pd.Series of np.ndarray of np.float64
        Array of ratios for:
        Mean over threshold of the last but one alert
        Mean over threshold of the last alert
        Measurement over threshold of the last alert
        All the default values are set to -1 as it is an unphysical value

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> import pandas as pd
    >>> from fink_utils.spark.utils import concat_col
    >>> import pyspark.sql.functions as F
    >>> from fink_science.ztf.standardized_flux.processor import standardized_flux

    >>> parDF = spark.read.parquet(ztf_alert_sample)

    # Required alert columns
    >>> what = [
    ...     'distnr',
    ...     'magpsf',
    ...     'sigmapsf',
    ...     'magnr',
    ...     'sigmagnr',
    ...     'isdiffpos',
    ...     'fid',
    ...     'jd'
    ... ]

    # Concatenation
    >>> prefix = 'c'
    >>> for key in what:
    ...     parDF = concat_col(parDF, colname=key, prefix=prefix)

    # Preliminary module run
    >>> args = [
    ...     'candid',
    ...     'objectId',
    ...     'cdistnr',
    ...     'cmagpsf',
    ...     'csigmapsf',
    ...     'cmagnr',
    ...     'csigmagnr',
    ...     'cisdiffpos',
    ...     'cfid',
    ...     'cjd'
    ... ]
    >>> parDF = parDF.withColumn(
    ...     'container',
    ...     standardized_flux(*args)
    ... )
    >>> parDF = parDF.withColumn(
    ...     'cstd_flux',
    ...     parDF['container'].getItem('flux')
    ... )
    >>> parDF = parDF.withColumn(
    ...     'csigma_std_flux',
    ...     parDF['container'].getItem('sigma')
    ... )

    # Drop temporary columns
    >>> what_prefix = [prefix + key for key in what]
    >>> parDF = parDF.drop('container')

    # Test the module
    >>> args = ['candid', 'objectId', 'cstd_flux', 'cjd']
    >>> parDF = parDF.withColumn('blazar_stats', quiescent_state(*args))

    # Test
    >>> pdf = parDF.select([F.col('blazar_stats').getItem('m0').alias("m0"), F.col('blazar_stats').getItem('m1').alias("m1"), F.col('blazar_stats').getItem('m2').alias("m2")]).toPandas()
    >>> (pdf.sum(axis=1) == -3).sum()
    320
    """
    path = os.path.dirname(os.path.abspath(__file__))
    CTAO_PATH = os.path.join(path, "data/catalogs")
    CTAO_filename = "CTAO_blazars_ztf_dr{}.parquet".format(RELEASE)
    CTAO_blazar = pd.read_parquet(os.path.join(CTAO_PATH, CTAO_filename))

    pdf = pd.DataFrame({
        "candid": candid,
        "objectId": objectId,
        "cstd_flux": cstd_flux,
        "cjd": cjd,
    })
    out = []
    for candid_ in pdf["candid"]:
        tmp = pdf[pdf["candid"] == candid_]
        if len(tmp["cstd_flux"].to_numpy()[0]) == 0:
            out.append({k: -1 for k in BLAZAR_COLS})
            continue
        sub = pd.DataFrame({
            "candid": tmp["candid"].to_numpy()[0],
            "objectId": tmp["objectId"].to_numpy()[0],
            "cstd_flux": tmp["cstd_flux"].to_numpy()[0],
            "cjd": tmp["cjd"].to_numpy()[0],
        })
        dic = {k: v for k, v in zip(BLAZAR_COLS, quiescent_state_(sub, CTAO_blazar))}  # noqa: C416
        out.append(dic)

    return pd.Series(out)


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()
    path = os.path.join(os.path.dirname(__file__), "data/alerts")
    filename = "CTAO_blazar_datatest_v20-12-24.parquet"
    ztf_alert_sample = "file://{}/{}".format(path, filename)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
