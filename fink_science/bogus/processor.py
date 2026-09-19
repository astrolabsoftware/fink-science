# Copyright 2025 AstroLab Software
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
from line_profiler import profile

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import IntegerType

from fink_science import __file__
import os

import pandas as pd
import numpy as np

from fink_science.tester import spark_unit_tests

@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
@profile
def ztf_bogus(rb, nbad):
    """ Determine if an alert is a potential bogus

    Parameters
    ----------
    rb: Spark DataFrame Column
        Real-bogus score calculated by ZTF
    nbad: Spark DataFrame Column
        Number of prior-tagged bad pixels in a 5 x 5 pixel stamp

    Returns
    ----------
    out: integer
        1 if the alert is likely a bogus
        0 if the alert is likely not a bogus

    Examples
    ----------
    >>> df = spark.read.load(ztf_alert_sample)
    >>> args = ['candidate.rb', 'candidate.nbad']
    >>> df = df.withColumn('bogusness', ztf_bogus(*args))

    >>> df.filter(df['bogusness'] == 0).count()
    3

    >>> df.filter(df['bogusness'] == 1).count()
    3
    """
    # All non-bogus
    flags = np.zeros_like(nbad.to_numpy(), dtype=int)

    # Conditions for being a bogus
    f0 = rb >= 0.55
    f1 = nbad == 0
    flags[f0 & f1] = 1

    return pd.Series(flags)


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)
    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
