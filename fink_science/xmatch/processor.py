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

from astropy.coordinates import SkyCoord
from astropy import units as u

import os
import pandas as pd
import numpy as np

from fink_science.xmatch.classification import cross_match_alerts_raw
from fink_science.xmatch.classification import extract_vsx, extract_gcvs
from fink_science.tester import spark_unit_tests
from fink_science import __file__

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

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def crossmatch_other_catalog(candid, ra, dec, catalog_name):
    """ Crossmatch alerts with user-defined catalogs

    Currently supporting:
    - GCVS
    - VSX

    Parameters
    ----------
    candid: long
        ZTF candidate ID
    ra: float
        ZTF Right ascension
    dec: float
        ZTF declinations
    catalog_name: str
        Name of the catalog to use. currently supported: gcvs, vsx

    Returns
    ----------
    type: str
        Object type from the catalog. `Unknown` if no match.

    Examples
    ----------
    >>> from pyspark.sql.functions import lit

    Simulate fake data
    >>> ra = [26.8566983, 101.3520545]
    >>> dec = [-26.9677112, 24.5421872]
    >>> id = ["1", "2"]

    Wrap data into a Spark DataFrame
    >>> rdd = spark.sparkContext.parallelize(zip(id, ra, dec))
    >>> df = rdd.toDF(['id', 'ra', 'dec'])
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+
    | id|         ra|        dec|
    +---+-----------+-----------+
    |  1| 26.8566983|-26.9677112|
    |  2|101.3520545| 24.5421872|
    +---+-----------+-----------+
    <BLANKLINE>

    Test the processor by adding a new column with the result of the xmatch
    >>> df.withColumn(
    ...     'gcvs',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('gcvs'))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+-------+
    | id|         ra|        dec|   gcvs|
    +---+-----------+-----------+-------+
    |  1| 26.8566983|-26.9677112|Unknown|
    |  2|101.3520545| 24.5421872|     RR|
    +---+-----------+-----------+-------+
    <BLANKLINE>

    >>> df = df.withColumn(
    ...     'vsx',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('vsx'))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+----+
    | id|         ra|        dec| vsx|
    +---+-----------+-----------+----+
    |  1| 26.8566983|-26.9677112|MISC|
    |  2|101.3520545| 24.5421872|RRAB|
    +---+-----------+-----------+----+
    <BLANKLINE>
    """
    pdf = pd.DataFrame(
        {
            'ra': ra.values,
            'dec': dec.values,
            'candid': range(len(ra))
        }
    )

    curdir = os.path.dirname(os.path.abspath(__file__))
    if catalog_name.values[0] == 'gcvs':
        catalog = curdir + '/data/catalogs/gcvs.parquet'
        ra2, dec2, type2 = extract_gcvs(catalog)
    elif catalog_name.values[0] == 'vsx':
        catalog = curdir + '/data/catalogs/vsx.parquet'
        ra2, dec2, type2 = extract_vsx(catalog)

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.values, dtype=np.float) * u.degree,
        dec=np.array(dec.values, dtype=np.float) * u.degree
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.values, dtype=np.float) * u.degree,
        dec=np.array(dec2.values, dtype=np.float) * u.degree
    )

    # cross-match
    idx, d2d, d3d = catalog_other.match_to_catalog_sky(catalog_ztf)

    # set separation length
    sep_constraint = d2d.degree < 1.5 / 3600

    catalog_matches = np.unique(pdf['candid'].values[idx[sep_constraint]])

    # identify position of matches in the input dataframe
    pdf_matches = pd.DataFrame(
        {
            'candid': np.array(catalog_matches, dtype=np.int64),
            'match': True
        }
    )
    pdf_merge = pd.merge(pdf, pdf_matches, how='left', on='candid')

    m = pdf_merge['match'].apply(lambda x: x is True)

    # Now get types for these
    catalog_ztf_merge = SkyCoord(
        ra=np.array(pdf_merge.loc[m, 'ra'].values, dtype=np.float) * u.degree,
        dec=np.array(pdf_merge.loc[m, 'dec'].values, dtype=np.float) * u.degree
    )

    # cross-match
    idx2, d2d2, d3d2 = catalog_ztf_merge.match_to_catalog_sky(catalog_other)

    pdf_merge['Type'] = 'Unknown'
    pdf_merge.loc[m, 'Type'] = [
        str(i).strip() for i in type2.astype(str).values[idx2]
    ]

    return pdf_merge['Type']


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    spark_unit_tests(globals())
