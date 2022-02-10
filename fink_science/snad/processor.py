import logging
import os

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType, DoubleType, ArrayType

import pandas as pd
import numpy as np
import light_curve as lc

from fink_science import __file__
from fink_science.tester import spark_unit_tests


logger = logging.getLogger(__name__)


def create_extractor():
    """
    Features definition - https://arxiv.org/pdf/2012.01419.pdf#section.A1
    """

    return lc.Extractor(
        lc.Mean(),                                     # A1.0.2  Mean, A2.0.12  Mean
        lc.WeightedMean(),                             # A1.0.3  Weighted Mean, A2.0.29  Weighted mean
        lc.StandardDeviation(),                        # A1.0.4  Standard Deviation, A2.0.27  Standard Deviation
        lc.Median(),                                   # A1.0.7  Median

        lc.Amplitude(),                                # A2.0.1  Amplitude
        lc.BeyondNStd(nstd=1),                         # A2.0.2  Beyond𝑛Std
        lc.Cusum(),                                    # A2.0.3  Cusum
        lc.InterPercentileRange(quantile=0.1),         # A2.0.6  Inter-percentile range
        lc.Kurtosis(),                                 # A2.0.7  Kurtosis
        lc.LinearTrend(),                              # A2.0.8  Linear Trend
        lc.LinearFit(),                                # A2.0.9  Linear Fit
        lc.MagnitudePercentageRatio(                   # A2.0.10  Magnitude Percentage Ratio ; 0.4, 0.05 and 0.2, 0.1 are 'default' values
            quantile_numerator=0.4,
            quantile_denominator=0.05,
        ),
        lc.MagnitudePercentageRatio(
            quantile_numerator=0.2,
            quantile_denominator=0.1,
        ),
        lc.MaximumSlope(),                             # A2.0.11  Maximum Slope
        lc.MedianAbsoluteDeviation(),                  # A2.0.13  Median Absolute Deviation
        lc.MedianBufferRangePercentage(quantile=0.1),  # A2.0.14  Median Buffer Range Percentage
        lc.PercentAmplitude(),                         # A2.0.15  Percent Amplitude
        lc.MeanVariance(),                             # 
        lc.AndersonDarlingNormal(),                    # 
        lc.ReducedChi2(),                              # A2.0.25  Reduced𝜒2
        lc.Skew(),                                     # A2.0.26  Skew
        lc.StetsonK(),                                 # A2.0.28  Stetson𝐾

    )


# 'lc.Extrator' can not be pickled, and thus needs to be created inside the udf,
# but we also need the list of names outside the udf
column_names = create_extractor().names
columns_count = len(column_names)


@pandas_udf(ArrayType(DoubleType()), PandasUDFType.SCALAR)
def extract_features_snad(
    arr_magpsf,
    arr_jd,
    arr_sigmapsf,
    arr_cfid,
    arr_oId
) -> pd.DataFrame:
    """ Returns many features, extracted from measurment's using light_curve package (https://github.com/light-curve/light-curve-python).
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

    Returns
    ----------
    out: list
        Returns list of lists. Each "inner" list consists of many light curve features for each filter.

    Examples
    ---------
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> snad_base_col = 'lc_features'
    >>> df = df.withColumn(snad_base_col, extract_features_snad(*what_prefix))
    >>> for index, name in enumerate(column_names):
    ...     df = df.withColumn("fid_1_" + name, df[snad_base_col][index])
    ...     df = df.withColumn("fid_2_" + name, df[snad_base_col][index + columns_count])

    >>> df = df.drop(snad_base_col, *what_prefix)

    # TODO: better assertions
    >>> df.filter(df['fid_1_mean'] < 1e-1).count()
    0
    """

    results = []
    extractor = create_extractor()

    for magpsf, jd, sigmapsf, cfid, oId in zip(arr_magpsf, arr_jd, arr_sigmapsf, arr_cfid, arr_oId):
        # Select only valid measurements (not upper limits)
        maskNotNone = magpsf == magpsf

        magpsf = magpsf.astype("float64")
        jd = jd.astype("float64")
        sigmapsf = jd.astype("float64")

        nans = np.isnan(magpsf) | np.isnan(sigmapsf)
        magpsf = magpsf[~nans & maskNotNone]
        sigmapsf = sigmapsf[~nans & maskNotNone]
        jd = jd[~nans & maskNotNone]
        cfid = cfid[~nans & maskNotNone]

        passband_one = cfid == 1
        passband_two = cfid == 2   

        full_result = []
        for passband in (passband_one, passband_two):
            try:
                result = extractor(jd[passband], magpsf[passband], sigmapsf[passband], fill_value=np.nan)
            except ValueError as err:
                full_result = None
                # log if known error, then skip
                if err.args[0] == "t must be in ascending order":
                    logger.error(f"Unordered jd for {oId} in processor '{__file__}/{extract_features_snad.__name__}'")
                else:
                    logger.exception(f"Unknown exception in processor '{__file__}/{extract_features_snad.__name__}'")
                break
            except Exception as err:
                full_result = None
                logger.exception(f"Unknown exception in processor '{__file__}/{extract_features_snad.__name__}'")
                break
            full_result.extend(result)

        results.append(full_result)

    return pd.Series(results)


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample
    
    # Run the test suite
    spark_unit_tests(globs)
