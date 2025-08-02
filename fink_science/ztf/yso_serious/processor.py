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
        """
        Calculates the r2 statistics. Goal is to check goodness of fit.

        Parameters
        ----------
        x: Spark DataFrame Column
           observed jd
        y: Spark DataFrame Column
            observed psf magnitude

        Returns
        -------
        out: Spark DataFrame Column
            R2 statistics        
        """
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

        out = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        return out

    return pd.Series([r2_one(x, y) for x, y in zip(xs, ys)])

@F.pandas_udf(FloatType())
@profile
def filter_spicy_trend(spicy_class: pd.Series, linear_fit_slope: pd.Series,
                   cjd_r: pd.Series, cmagpsf_r: pd.Series, n_rband:pd.Series) -> pd.Series:
    """ Filter objects in spicy with significant trend in the last 30 days.

    Notes
    -----
    First iteration cuts: 
        min npoints in r-band = 5
        r2 > 0.6
        linear_fit_slope > 0.025
    
    Parameters
    ----------
    spicy_class: Spark DataFrame Column
        Class of the xmatched spicy object
    linear_fit_slope: Spark DataFrame Column
        from lc_features_r
    cjd_r: Spark DataFrame Column
        time of observation, current alert + history - r-band only
    cmagpsf_r: Spark DataFrame Column
        observed psf magnitude, current alert + history - r-band only
    n_rband: Spark DataFrame Column
        number of observations in the r-band

    Returns
    ----------
    r2: pd.Series
        Difference magnitude between last 2 measurements

    Examples
    ----------
    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.getOrCreate()
    
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    
    >>> df = spark.read.format('parquet').load(test_yso_cuts)
    >>> df.count()
    26

    # Required alert columns
    >>> what = ['jd', 'magpsf', 'fid']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> df = concat_col(df, 'jd', prefix=prefix)
    >>> df = concat_col(df, 'magpsf', prefix=prefix)
    >>> df = concat_col(df, 'fid', prefix=prefix)

    >>> df = df.withColumn("z", F.arrays_zip("cfid", "cmagpsf","cjd"))
    >>> df = df.withColumn("z2", F.expr("filter(z, x -> x.cfid = 2 AND x.cmagpsf IS NOT NULL AND x.cjd IS NOT NULL)"))
    >>> df = df.withColumn("cmagpsf_masked", F.expr("transform(z2, x -> x.cmagpsf)"))
    >>> df = df.withColumn("cjd_masked",     F.expr("transform(z2, x -> x.cjd)"))
    >>> df = df.withColumn('n_rband', F.size('cmagpsf_masked')).drop("z", "z2")
    
    # apply the science module
    >>> df = df.withColumn('spicy_trend',filter_spicy_trend('spicy_class','lc_features_r.linear_fit_slope',  'cjd_masked', 'cmagpsf_masked', 'n_rband'))

    # Count the number of survivors
    >>> int(df.select(F.sum(F.col("spicy_trend"))).collect()[0][0])
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

    return mask.astype(float)



if __name__ == "__main__":
    """ Execute the test suite """
    
    globs = globals()
    path = os.path.dirname(__file__)
    sample_file = (
        "./fink_science/data/alerts/test_yso_cuts.parquet"
    )
    globs["test_yso_cuts"] = sample_file
    spark_unit_tests(globs)
