# Copyright 2020 AstroLab Software
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
# See the License for the specific language governing permissions and
# limitations under the License.
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

import pandas as pd

from utilities import compute_delta

@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def deltamaglatest(magpsf) -> pd.Series:
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
    >>> df = spark.read.format('parquet').load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['magpsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # apply the science module
    >>> df = df.withColumn('deltamaglatest', deltamaglatest('cmagpsf'))

    # Drop temp columns
    >>> df = df.drop(*what_prefix)
    """
    delta_mag = magpsf.apply(compute_delta)

    return delta_mag