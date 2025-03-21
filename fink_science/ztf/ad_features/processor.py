# Copyright 2020-2025 AstroLab Software
# Author: Igor Beschastnov, Julien Peloton
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

import logging
import os

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, IntegerType, MapType, StructType, StructField

import pandas as pd
import numpy as np
import light_curve as lc

from fink_science import __file__
from fink_science.tester import spark_unit_tests

from fink_utils.photometry.utils import is_source_behind
from fink_utils.photometry.vect_conversion import vect_dc_mag

logger = logging.getLogger(__name__)


def create_extractor():
    """Features definition - https://arxiv.org/pdf/2012.01419.pdf#section.A1"""
    return lc.Extractor(
        lc.Mean(),  # A1.0.2  Mean, A2.0.12  Mean
        lc.WeightedMean(),  # A1.0.3  Weighted Mean, A2.0.29  Weighted mean
        lc.StandardDeviation(),  # A1.0.4  Standard Deviation, A2.0.27  Standard Deviation
        lc.Median(),  # A1.0.7  Median
        lc.Amplitude(),  # A2.0.1  Amplitude
        lc.BeyondNStd(nstd=1),  # A2.0.2  Beyond𝑛Std
        lc.Cusum(),  # A2.0.3  Cusum
        lc.InterPercentileRange(quantile=0.1),  # A2.0.6  Inter-percentile range
        lc.Kurtosis(),  # A2.0.7  Kurtosis
        lc.LinearTrend(),  # A2.0.8  Linear Trend
        lc.LinearFit(),  # A2.0.9  Linear Fit
        lc.MagnitudePercentageRatio(  # A2.0.10  Magnitude Percentage Ratio ; 0.4, 0.05 and 0.2, 0.1 are 'default' values
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
        lc.MeanVariance(),  #
        lc.AndersonDarlingNormal(),  #
        lc.ReducedChi2(),  # A2.0.25  Reduced𝜒2
        lc.Skew(),  # A2.0.26  Skew
        lc.StetsonK(),  # A2.0.28  Stetson𝐾
    )


# 'lc.Extrator' can not be pickled, and thus needs to be created inside the udf,
# but we also need the list of names outside the udf
FEATURES_COLS = create_extractor().names
columns_count = len(FEATURES_COLS)


@profile
def extract_features_ad_raw(
    magpsf, jd, sigmapsf, cfid, oId, distnr, magnr=None, sigmagnr=None, isdiffpos=None
) -> pd.Series:
    """Returns features extracted from measurments using light_curve package

    Notes
    -----
    See https://github.com/light-curve/light-curve-python.
    Reference - https://arxiv.org/pdf/2012.01419.pdf#section.A1

    Parameters
    ----------
    jd: Spark DataFrame Column
        Julian date
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    fid: Spark DataFrame Column
        Filter IDs (int)
    oId: Spark DataFrame Column
        Object IDs (str)
    distnr: float
        distance to nearest source in reference image PSF-catalog within 30 arcsec [pixels]
    magnr,sigmagnr
        magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    isdiffpos
        t or 1 => candidate is from positive (sci minus ref) subtraction
        f or 0 => candidate is from negative (ref minus sci) subtraction

    Returns
    -------
    out: dict
        Returns dict of dict. Keys of first dict - filters (fid), keys of inner dicts - names of features.

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid', 'distnr', 'magnr', 'sigmagnr', 'isdiffpos']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> cols = ['cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId', 'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']
    >>> df = df.withColumn('lc_features', extract_features_ad(*cols))

    >>> for row in df.take(10):
    ...    assert len(row['lc_features']) == len(np.unique(row['cfid']))
    ...    assert len(row['lc_features'][1]) == 26

    # check the processing on i-band
    >>> df = spark.read.format('parquet').load(ztf_alert_with_i_band)

    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid', 'distnr', 'magnr', 'sigmagnr', 'isdiffpos']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> cols = ['cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId', 'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']
    >>> df = df.withColumn('lc_features', extract_features_ad(*cols))

    >>> for row in df.take(10):
    ...    assert len(row['lc_features']) == len(np.unique(row['cfid'])) - 1
    ...    assert 3 not in row['lc_features'].keys()
    """
    cfid = np.asarray(cfid, "int32")
    magpsf = np.asarray(magpsf, "float64")
    jd = np.asarray(jd, "float64")
    sigmapsf = np.asarray(sigmapsf, "float64")
    distnr = np.asarray(distnr, "float64")

    extractor = create_extractor()

    passbands = np.unique(cfid)

    # keep only g and/or r -- see #393
    maskFilter = passbands <= 2
    passbands = passbands[maskFilter]

    # Select only valid measurements (not upper limits)
    maskNotNone = magpsf == magpsf
    mask = ~(np.isnan(magpsf) | np.isnan(sigmapsf)) & maskNotNone

    magpsf = magpsf[mask]
    sigmapsf = sigmapsf[mask]

    # Compute DC mag if need be
    flag = is_source_behind(distnr[mask])
    maskDC = np.ones(len(magpsf), dtype=bool)
    if np.sum(flag) > 0:
        magnr = np.asarray(magnr, "float64")[mask]
        sigmagnr = np.asarray(sigmagnr, "float64")[mask]
        isdiffpos = np.asarray(isdiffpos, "str")[mask]

        # in-place replacement
        magpsf, sigmapsf = vect_dc_mag(magpsf, sigmapsf, magnr, sigmagnr, isdiffpos)
        maskDC = ~(np.isnan(magpsf) | np.isnan(sigmapsf))
        magpsf = magpsf[maskDC]
        sigmapsf = sigmapsf[maskDC]

    jd = jd[mask][maskDC]
    cfid = cfid[mask][maskDC]

    sub = pd.DataFrame({"magpsf": magpsf, "sigmapsf": sigmapsf, "jd": jd, "cfid": cfid})

    sub = sub.sort_values("jd", ascending=True)

    full_result = {}
    for passband_id in passbands:
        passband = sub["cfid"].to_numpy() == passband_id
        try:
            result = extractor(
                sub["jd"].to_numpy()[passband],
                sub["magpsf"].to_numpy()[passband],
                sub["sigmapsf"].to_numpy()[passband],
                fill_value=np.nan,
            )
        except ValueError as err:
            # log if known error, then skip
            if err.args[0] == "t must be in ascending order":
                logger.error(
                    f"Unordered jd for {oId} in processor '{__file__}/{extract_features_ad.__name__}'"
                )
            else:
                logger.exception(
                    f"Unknown exception for {oId} in processor '{__file__}/{extract_features_ad.__name__}'"
                )
            continue
        except Exception:
            logger.exception(
                f"Unknown exception for {oId} in processor '{__file__}/{extract_features_ad.__name__}'"
            )
            continue
        full_result[int(passband_id)] = dict(
            zip(FEATURES_COLS, [float(v) for v in result])
        )

    return full_result


extract_features_ad = udf(
    f=extract_features_ad_raw,
    returnType=MapType(
        IntegerType(),  # passband_id
        StructType([  # features name -> value
            StructField(name, DoubleType(), True) for name in FEATURES_COLS
        ]),
    ),
)


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample
    del globs["extract_features_ad_raw"]

    ztf_alert_with_i_band = (
        "file://{}/data/alerts/20240606_iband_history.parquet".format(path)
    )
    globs["ztf_alert_with_i_band"] = ztf_alert_with_i_band

    # Run the test suite
    spark_unit_tests(globs)
