# Copyright 2019-2021 AstroLab Software
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
from pyspark.sql.types import StringType

import pandas as pd
import numpy as np

from fink_science.xmatch.classification import cross_match_alerts_raw
from fink_science.tester import spark_unit_tests

from typing import Any

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def cdsxmatch(objectId: Any, ra: Any, dec: Any) -> pd.Series:
    """ Query the CDSXmatch service to find identified objects
    in alerts. The catalog queried is the SIMBAD bibliographical database.

    I/O specifically designed for use as `pandas_udf` in `select` or
    `withColumn` dataframe methods

    The user will create a new processing function with his/her needs the
    following way:

    1) Define the input entry column. These must be `candidate` entries.
    2) Update the logic inside the function. The idea is to
        apply conditions based on the values of the columns.
    3) Return a column with added value after processing

    Parameters
    ----------
    objectId: list of str or Spark DataFrame Column of str
        List containing object ids (custom)
    ra: list of float or Spark DataFrame Column of float
        List containing object ra coordinates
    dec: list of float or Spark DataFrame Column of float
        List containing object dec coordinates

    Returns
    ----------
    out: pandas.Series of string
        Return a Pandas DataFrame with the type of object found in Simbad.
        If the object is not found in Simbad, the type is
        marked as Unknown. In the case several objects match
        the centroid of the alert, only the closest is returned.
        If the request Failed (no match at all), return Column of Fail.

    Examples
    -----------
    Simulate fake data
    >>> ra = [26.8566983, 26.24497]
    >>> dec = [-26.9677112, -26.7569436]
    >>> id = ["1", "2"]

    Wrap data into a Spark DataFrame
    >>> rdd = spark.sparkContext.parallelize(zip(id, ra, dec))
    >>> df = rdd.toDF(['id', 'ra', 'dec'])
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+----------+-----------+
    | id|        ra|        dec|
    +---+----------+-----------+
    |  1|26.8566983|-26.9677112|
    |  2|  26.24497|-26.7569436|
    +---+----------+-----------+
    <BLANKLINE>

    Test the processor by adding a new column with the result of the xmatch
    >>> df = df.withColumn(
    ...     'cdsxmatch', cdsxmatch(df['id'], df['ra'], df['dec']))
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+----------+-----------+---------+
    | id|        ra|        dec|cdsxmatch|
    +---+----------+-----------+---------+
    |  1|26.8566983|-26.9677112|     Star|
    |  2|  26.24497|-26.7569436|  Unknown|
    +---+----------+-----------+---------+
    <BLANKLINE>
    """
    # your logic goes here
    matches = cross_match_alerts_raw(
        objectId.values, ra.values, dec.values)

    # For regular alerts, the number of matches is always non-zero as
    # alerts with no counterpart will be labeled as Unknown.
    # If cross_match_alerts_raw returns a zero-length list of matches, it is
    # a sign of a problem (logged).
    if len(matches) > 0:
        # (objectId, ra, dec, name, type)
        # return only the type.
        names = np.transpose(matches)[-1]
    else:
        # Tag as Fail if the request failed.
        names = ["Fail"] * len(objectId)
    return pd.Series(names)


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    spark_unit_tests(globals())
