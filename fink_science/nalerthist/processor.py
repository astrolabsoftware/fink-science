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
from pyspark.sql.types import IntegerType

import pandas as pd
import numpy as np

import os

from fink_science import __file__

from fink_science.tester import spark_unit_tests

@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def nalerthist(magpsf) -> pd.Series:
    """ Compute the number of detections contained in the alert (current+history)
    Upper limits are not counted.

    Parameters
    ----------
    magpsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry

    Returns
    ----------
    probabilities: 1D np.array of int
        Number of detections contained in the alert (current+history)
        Upper limits are not counted.

    Examples
    ----------
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Append temp columns with historical + current measurements
    >>> df = concat_col(df, 'magpsf', prefix='c')

    # Perform the fit + classification (default model)
    >>> df = df.withColumn('nalerthist', nalerthist(df['cmagpsf']))

    >>> df.agg({"nalerthist": "min"}).collect()[0][0]
    1

    >>> df.agg({"nalerthist": "max"}).collect()[0][0]
    9
    """
    nalerthist = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x)))
    return nalerthist


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/alerts.parquet'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
