# Copyright 2020-2023 AstroLab Software
# Author: Julien Peloton
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
# See the License for the specififrom pyspark.sql import SparkSession

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType, StringType
from line_profiler import profile
import pyspark.sql.functions as F

import pandas as pd
import numpy as np
import os

from fink_science.tester import spark_unit_tests

@pandas_udf(FloatType(), PandasUDFType.SCALAR)
@profile
def r2_from_arrays(xs: pd.Series, ys: pd.Series) -> pd.Series:
    def r2_one(x, y):
        if x is None or y is None:
            return np.nan
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 5 or y.size < 5 or x.size != y.size:
            return np.nan

        x_mean = x.mean(); y_mean = y.mean()
        sxx = np.sum((x - x_mean) ** 2)
        sxy = np.sum((x - x_mean) * (y - y_mean))
        if sxx == 0:
            return np.nan
        beta1 = sxy / sxx
        beta0 = y_mean - beta1 * x_mean
        y_hat = beta0 + beta1 * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return pd.Series([r2_one(x, y) for x, y in zip(xs, ys)])

@F.pandas_udf(ArrayType(FloatType()))
@profile
def filter_spicy_trend(spicy_class: pd.Series, linear_fit_slope: pd.Series,
                   cjd_r: pd.Series, cmagpsf_r: pd.Series, n_rband:pd.Series) -> pd.Series:
    """ Compute the change in magnitude between the 
    2 latest magnitudes.

    Parameters
    ----------
    magpsf: Spark DataFrame Column
        Magnitude from PSF-fit photometry

    Returns
    ----------
    delta_mag: pd.Series
        Difference magnitude between last 2 measurements

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
    >>> df = spark.read.format('parquet').load(test_yso_cut.parquet)
    >>> df.count()
    26

    # Required alert columns
    >>> what = ['jd', 'magpsf', 'fid']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
            df = concat_col(df, colname, prefix=prefix)

    >>> df2 = (
        df.withColumn("z", F.arrays_zip("cfid", "cmagpsf", "cjd"))
        # keep only positions r-band (cfid == 2)
        .withColumn("z2", F.expr("filter(z, x -> x.cfid = 2 AND x.cmagpsf IS NOT NULL AND x.cjd IS NOT NULL)"))
        # project masked arrays back out
        .withColumn("cmagpsf_masked", F.expr("transform(z2, x -> x.cmagpsf)"))
        .withColumn("cjd_masked",     F.expr("transform(z2, x -> x.cjd)"))
        # count number of points in the r-band
        .withColumn('n_rband', F.size('cmagpsf_masked'))
        .drop("z", "z2")
    )
    
    # apply the science module
    >>> df2 = df2.withColumn('spicy_trend', 
                            filter_spicy_trend('spicy_class', 'lc_features_r.linear_fit_slope',
                            'cjd_masked', 'cmagpsf_masked', 'n_rband'))

    # Count the number of survivors
    >>> df3 = df2.filter(F.col('spicy_trend') == 1.0)
    >>> df3.count()
    6
    """
    
    slope_lim = 0.025   # minimum slope threshold
    npoints = 5         # minimum required number of points
    r2_lim = 0.6        # minimum required r2
    
    # select spicy objecs
    mask_spicy = (spicy_class != "Unknown")

    # select spicy objects which respect the slope threshold
    mask_slope = mask_spicy & (linear_fit_slope.abs() > slope_lim)

    # select objects with minimum number of points
    mask_points = mask_slope & (n_rband >= npoints)

    use_magpsf = cmagpsf_r.where(mask_points, np.nan)
    use_jd = cjd_r.where(mask_points, np.nan)

    r2 = r2_from_arrays(use_jd, use_magpsf)
    mask = r2 > r2_lim

    return pd.Series(mask)



if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    test_yso_cut = "file://{}/data/alerts".format(path)
    globs["test_yso_cut"] = test_yso_cut

    # Run the test suite
    spark_unit_tests(globs)
