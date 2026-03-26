# Copyright 2026 AstroLab Software
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
from fink_science.ztf.blazar_extreme_state.utils import (
    extreme_state_,
    get_ztf_dr_data,
    from_mag_to_flux,
    standardise_lc,
    compute_quantile,
)

from fink_science.tester import spark_unit_tests
from fink_science import __file__
import os


# ====================================
# Constants required by the processor
# ====================================

# Latest catalog version name
CATALOG_TAG = "23.v03_2026"

# New columns to be produced and added to the scheme
BLAZAR_LOW_COLS = ["instantness_low", "robustness_low"]
INST_LOW_TAG, ROB_LOW_TAG = BLAZAR_LOW_COLS
BLAZAR_HIGH_COLS = ["instantness_high", "robustness_high"]
INST_HIGH_TAG, ROB_HIGH_TAG = BLAZAR_LOW_COLS
CDF_COL = ["cdf_quantile"]
CDF_TAG = CDF_COL[0]

# Integration periods for the computation
# of the fluence in the robustness criterion
INTEGRATION_PERIOD_LOW = 30
INTEGRATION_PERIOD_HIGH = 30

# Radius for the cone search in the Data Release
RADIUS = 2

# Parameters needed for the standardisation of the light curve
TIME_BINNING = 7  # days
DT_CONCOMITANCE = 0 + 1 / 24 + 0 / 60  # days, hours, minutes


# ==========
# Processor
# ==========


@pandas_udf(MapType(StringType(), FloatType()))
@profile
def extreme_state(
    candid: pd.Series,
    objectId: pd.Series,
    cstd_flux: pd.Series,
    cjd: pd.Series,
    cra: pd.Series,
    cdec: pd.Series,
) -> pd.Series:
    """
    Returns an array containing extreme state blazar features.

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf, cmagnr, cra, cdec,
        csigmagnr, cisdiffpos, cfid, cjd, cstd_flux, csigma_std_flux
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.

    Returns
    -------
    out: pd.Series of np.ndarray of np.float64
        Array of ratios for:\n
        \t-Measurement over threshold of the last alert for low states.\n
        \t-Fluence over threshold of the last alerts for low states.\n
        \t-Measurement over threshold of the last alert for high states.\n
        \t-Fluence over threshold of the last alerts for high states.\n
        \t-Corresponding quantile of hte last measurement on the CDF.\n
        All the default values are set to -1 as it is an unphysical value.

    Notes
    -----
    Features are:
    \t-``instantness_low``: Measurement over threshold of the last alert for
    low states.\n
    \t-``robustness_low``: Fluence over threshold of the last alerts for low
    states.\n
    \t-``instantness_high``: Measurement over threshold of the last alert for
    high states.\n
    \t-``robustness_high``: Fluence over threshold of the last alerts for high
    states.\n
    \t-``cdf_quantile``: Corresponding quantile of hte last measurement on the
    CDF.\n

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
    ...     'jd',
    ...     'ra',
    ...     'dec',
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
    ...     'cjd',
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
    >>> args = ['candid', 'objectId', 'cstd_flux', 'cjd', 'cra', 'cdec']
    >>> parDF = parDF.withColumn('blazar_stats', extreme_state(*args))

    # Test
    >>> pdf = parDF.select(
    ...     [F.col('blazar_stats').getItem(
    ...         'instantness_low'
    ...     ).alias("instantness_low"),
    ...     [F.col('blazar_stats').getItem(
    ...         'robustness_low'
    ...     ).alias("robustness_low"),
    ...     [F.col('blazar_stats').getItem(
    ...         'instantness_high'
    ...     ).alias("instantness_high"),
    ...     [F.col('blazar_stats').getItem(
    ...         'robustness_high'
    ...     ).alias("robustness_high"),
    ...     [F.col('blazar_stats').getItem(
    ...         'cdf_quantile'
    ...     ).alias("cdf_quantile")]
    ... ).toPandas()
    >>> (pdf[['instantness_low', 'robustness_low']].sum(axis=1) == -1).sum()
    322
    """
    # Load catalog
    path = os.path.dirname(os.path.abspath(__file__))
    CTAO_PATH = os.path.join(path, "data/catalogs")
    CTAO_filename = "CTAO_blazars_ztf_dr{}.parquet".format(CATALOG_TAG)
    CTAO_blazar = pd.read_parquet(os.path.join(CTAO_PATH, CTAO_filename))

    # Transform alert packet to pandas DataFrame
    pdf = pd.DataFrame({
        "candid": candid,
        "objectId": objectId,
        "cstd_flux": cstd_flux,
        "cjd": cjd,
        "cra": cra,
        "cdec": cdec,
    })
    out = []

    # Loop over all candidate ids
    for candid_ in pdf["candid"]:
        tmp = pdf[pdf["candid"] == candid_]

        # If no standardised flux has been calculated:
        if not len(tmp["cstd_flux"].to_numpy()[0]):
            out.append({k: -1. for k in BLAZAR_LOW_COLS + BLAZAR_HIGH_COLS + CDF_COL})
            continue

        # Else:
        sub = pd.DataFrame({
            "candid": tmp["candid"].to_numpy()[0],
            "objectId": tmp["objectId"].to_numpy()[0],
            "cstd_flux": tmp["cstd_flux"].to_numpy()[0],
            "cjd": tmp["cjd"].to_numpy()[0],
            "cra": tmp["cra"].to_numpy()[0],
            "cdec": tmp["cdec"].to_numpy()[0],
        })

        # Low state verification
        low_state_dic = dict(
            zip(
                BLAZAR_LOW_COLS,
                extreme_state_(
                    sub, CTAO_blazar, "low_threshold", INTEGRATION_PERIOD_LOW
                ),
            )
        )

        # High state verification
        high_state_dic = {k: -1 for k in BLAZAR_HIGH_COLS}
        if low_state_dic[INST_LOW_TAG] > 1 or low_state_dic[ROB_LOW_TAG] > 1:
            high_state_dic = dict(
                zip(
                    BLAZAR_HIGH_COLS,
                    extreme_state_(
                        sub, CTAO_blazar, "high_threshold", INTEGRATION_PERIOD_HIGH
                    ),
                )
            )

        # CDF computation
        cdf_dic = {CDF_TAG: -1}
        if (
            0 <= low_state_dic[INST_LOW_TAG] <= 1
            and 0 <= low_state_dic[ROB_LOW_TAG] <= 1
        ) or (high_state_dic[INST_HIGH_TAG] >= 1 and high_state_dic[ROB_HIGH_TAG] >= 1):
            measurement = sub["cstd_flux"].iloc[0]
            lc = get_ztf_dr_data(sub["cra"].mean(), sub["cdec"].mean(), RADIUS)
            lc = from_mag_to_flux(lc)
            lc = standardise_lc(sub, lc, CTAO_blazar)
            cdf_dic = {CDF_TAG: compute_quantile(lc, measurement)}

        out.append(low_state_dic | high_state_dic | cdf_dic)

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
