# Copyright 2019-2026 AstroLab Software
# Author: Timofei Pshenichnyy, Matwey V. Kornilov, Julien Peloton
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

import logging
import pandas as pd
import numpy as np
import light_curve as lc

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType, MapType, StructType, StructField

from fink_science.tester import spark_unit_tests
from fink_science import __file__


logger = logging.getLogger(__name__)

LSST_FILTER_LIST = ["u", "g", "r", "i", "z", "y"]


def create_extractor():
    """
    Features definition - identical to ZTF implementation for consistency.

    Reference: https://arxiv.org/pdf/2012.01419.pdf#section.A1
    Note: Features applied on Flux (nJy) instead of Magnitude.
    """
    return lc.Extractor(
        lc.Mean(),  # A1.0.2  Mean
        lc.WeightedMean(),  # A1.0.3  Weighted Mean
        lc.StandardDeviation(),  # A1.0.4  Standard Deviation
        lc.Median(),  # A1.0.7  Median
        lc.Amplitude(),  # A2.0.1  Amplitude
        lc.BeyondNStd(nstd=1),  # A2.0.2  Beyond n Std
        lc.Cusum(),  # A2.0.3  Cusum
        lc.InterPercentileRange(quantile=0.1),  # A2.0.6  Inter-percentile range
        lc.Kurtosis(),  # A2.0.7  Kurtosis
        lc.LinearTrend(),  # A2.0.8  Linear Trend
        lc.LinearFit(),  # A2.0.9  Linear Fit
        lc.MagnitudePercentageRatio(  # A2.0.10  Magnitude Percentage Ratio
            quantile_numerator=0.4,
            quantile_denominator=0.05,
        ),
        lc.MagnitudePercentageRatio(
            quantile_numerator=0.2,
            quantile_denominator=0.1,
        ),
        lc.MaximumSlope(),  # A2.0.11  Maximum Slope
        lc.MedianAbsoluteDeviation(),  # A2.0.13  Median Absolute Deviation
        lc.MedianBufferRangePercentage(
            quantile=0.1
        ),  # A2.0.14  Median Buffer Range Percentage
        lc.PercentAmplitude(),  # A2.0.15  Percent Amplitude
        lc.MeanVariance(),
        lc.AndersonDarlingNormal(),
        lc.ReducedChi2(),  # A2.0.25  Reduced Chi2
        lc.Skew(),  # A2.0.26  Skew
        lc.StetsonK(),  # A2.0.28  Stetson K
    )


# 'lc.Extractor' cannot be pickled, so we recreate it needed,
# but we define column names globally.
FEATURES_COLS = create_extractor().names


@profile
def extract_features_ad_rubin_raw(
    midpointMjdTai, psfFlux, psfFluxErr, band, diaObjectId
) -> pd.Series:
    """Returns features extracted from measurements using light_curve package for LSST alerts.

    Parameters
    ----------
    midpointMjdTai: Spark DataFrame Column
        MJD TAI times (vectors of floats)
    psfFlux: Spark DataFrame Column
        PSF Flux in nJy (vectors of floats)
    psfFluxErr: Spark DataFrame Column
        PSF Flux error in nJy (vectors of floats)
    band: Spark DataFrame Column
        Filter bands (vectors of strings: 'u', 'g', 'r', 'i', 'z', 'y')
    diaObjectId: Spark DataFrame Column
        Object IDs (vectors of str or int)

    Returns
    -------
    out: dict
        Returns dict of dict.
        Outer keys: filters (band name as single char),
        Inner keys: names of features.

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(rubin_alert_sample)

    # Required alert columns, concatenated with historical data
    >>> what = ['midpointMjdTai', 'psfFlux', 'psfFluxErr', 'band']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix, current='diaSource', history='prvDiaSources')

    >>> cols = [F.col(i) for i in what_prefix + ['diaObject.diaObjectId']]
    >>> df = df.withColumn('lc_features', extract_features_ad_rubin(*cols))

    >>> for row in df.take(10):
    ...    assert len(row['lc_features']) == len(np.unique(row['cband'])), len(np.unique(row['cband']))
    ...    for band_name in np.unique(row['cband']):
    ...        assert len(row['lc_features'][band_name]) == 26, len(row['lc_features'][band_name])
    """
    midpointMjdTai = np.asarray(midpointMjdTai, dtype=np.float64)
    psfFlux = np.asarray(psfFlux, dtype=np.float64)
    psfFluxErr = np.asarray(psfFluxErr, dtype=np.float64)
    band = np.asarray(band, dtype=str)

    extractor = create_extractor()

    try:
        df = pd.DataFrame({
            "time": midpointMjdTai,
            "flux": psfFlux,
            "err": psfFluxErr,
            "band": band,
        })
    except ValueError:
        logger.error(f"Array length mismatch for object {diaObjectId}")
        return {}

    # Remove NaNs
    mask = df[["time", "flux", "err"]].notna().all(axis=1)
    df = df[mask]

    if df.empty:
        return {}

    # Sort by time (crucial for some features like Cusum, LinearTrend)
    df = df.sort_values(
        "time"
    ).drop_duplicates(
        subset="time"
    )  # Just in case, we delete duplicates, in case there is the same problem as with ZTF.

    full_result = {}

    # Iterate over unique bands present in the alert history
    unique_bands = df["band"].unique()

    for filter_name in unique_bands:
        # Map string band to integer ID
        if filter_name not in LSST_FILTER_LIST:
            continue

        # Extract subset for this band
        sub = df[df["band"] == filter_name]

        if len(sub) < 1:
            continue

        try:
            result = extractor(
                sub["time"].to_numpy(),
                sub["flux"].to_numpy(),
                sub["err"].to_numpy(),
                sorted=True,  # We sorted beforehand
                fill_value=np.nan,
            )
        except ValueError as err:
            logger.error(f"Value Error for {diaObjectId} in band {filter_name}: {err}")
            continue
        except Exception as e:
            logger.exception(
                f"Unknown exception for {diaObjectId} in band {filter_name}: {e}"
            )
            continue

        # Pack into dictionary
        full_result[filter_name] = dict(zip(FEATURES_COLS, [float(v) for v in result]))

    return full_result


# Register the UDF
extract_features_ad_rubin = F.udf(
    f=extract_features_ad_rubin_raw,
    returnType=MapType(
        StringType(),  # passband
        StructType([  # features name -> value
            StructField(name, DoubleType(), True) for name in FEATURES_COLS
        ]),
    ),
)

if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()
    path = os.path.dirname(__file__)

    # from fink-alerts-schemas (see CI configuration)
    rubin_alert_sample = "file://{}/datasim/rubin_test_data_10_0.parquet".format(path)
    globs["rubin_alert_sample"] = rubin_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
