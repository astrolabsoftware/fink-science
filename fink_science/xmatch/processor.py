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
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType, MapType

from astropy.coordinates import SkyCoord
from astropy import units as u

import io
import os
import logging
import requests
import pandas as pd
import numpy as np

from fink_science.xmatch.utils import cross_match_astropy
from fink_science.xmatch.utils import generate_csv
from fink_science.xmatch.utils import extract_vsx, extract_gcvs
from fink_science.xmatch.utils import extract_3hsp, extract_4lac
from fink_science.xmatch.utils import extract_mangrove, MANGROVE_COLS
from fink_science.tester import spark_unit_tests
from fink_science import __file__

from typing import Any

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def cdsxmatch(objectId: Any, ra: Any, dec: Any, distmaxarcsec: float, extcatalog: str, cols: str) -> pd.Series:
    """ Query the CDSXmatch service to find identified objects
    in alerts. The catalog queried is the SIMBAD bibliographical database.

    I/O specifically designed for use as `pandas_udf` in `select` or
    `withColumn` dataframe methods

    Limitations known:
    - objectId should not be small integers.

    Parameters
    ----------
    objectId: list of str or Spark DataFrame Column of str
        List containing object ids (custom)
    ra: list of float or Spark DataFrame Column of float
        List containing object ra coordinates
    dec: list of float or Spark DataFrame Column of float
        List containing object dec coordinates
    distmaxarcsec: list of float or Spark DataFrame Column of float
        Cross-match radius in arcsecond
    extcatalog: list of str or Spark DataFrame Column of str
        Name of the external catalog in Vizier, or directly simbad.
    cols: list of str or Spark DataFrame Column of str
        Comma-separated column names to get from the external catalog

    Returns
    ----------
    out: pandas.Series of string
        Return Pandas DataFrame with a new single column
        containing comma-separated values of extra-columns.
        If the object is not found in Simbad, the type is
        marked as Unknown. In the case several objects match
        the centroid of the alert, only the closest is returned.
        If the request Failed (no match at all), return Column of Fail.

    Examples
    -----------
    Simulate fake data
    >>> ra = [26.8566983, 26.24497]
    >>> dec = [-26.9677112, -26.7569436]
    >>> id = ["a", "b"]

    Wrap data into a Spark DataFrame
    >>> rdd = spark.sparkContext.parallelize(zip(id, ra, dec))
    >>> df = rdd.toDF(['id', 'ra', 'dec'])
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+----------+-----------+
    | id|        ra|        dec|
    +---+----------+-----------+
    |  a|26.8566983|-26.9677112|
    |  b|  26.24497|-26.7569436|
    +---+----------+-----------+
    <BLANKLINE>

    Test the processor by adding a new column with the result of the xmatch
    >>> df = df.withColumn(
    ...     'cdsxmatch',
    ...     cdsxmatch(
    ...         df['id'], df['ra'], df['dec'],
    ...         F.lit(1.0), F.lit('simbad'), F.lit('main_type')))
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+----------+-----------+---------+
    | id|        ra|        dec|cdsxmatch|
    +---+----------+-----------+---------+
    |  a|26.8566983|-26.9677112|     LPV*|
    |  b|  26.24497|-26.7569436|  Unknown|
    +---+----------+-----------+---------+
    <BLANKLINE>
    """
    # If nothing
    if len(ra) == 0:
        return pd.Series([])

    # Catch TimeoutError and ConnectionError
    try:
        # Build a catalog of alert in a CSV-like string
        table_header = """ra_in,dec_in,objectId\n"""
        table = generate_csv(table_header, [ra, dec, objectId])

        # Send the request!
        r = requests.post(
            'http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
            data={
                'request': 'xmatch',
                'distMaxArcsec': distmaxarcsec.values[0],
                'selection': 'all',
                'RESPONSEFORMAT': 'csv',
                'cat2': extcatalog.values[0],
                'cols2': cols.values[0],
                'colRA1': 'ra_in',
                'colDec1': 'dec_in'},
            files={'cat1': table}
        )

        if r.status_code != 200:
            names = ["Fail {}".format(r.status_code)] * len(objectId)
            return pd.Series(names)
        else:
            cols = cols.values[0].split(',')
            pdf = pd.read_csv(io.BytesIO(r.content))

            if pdf.empty:
                name = ','.join(["Unknown"] * len(cols))
                names = [name] * len(objectId)
                return pd.Series(names)

            # join
            pdf_in = pd.DataFrame({'objectId_in': objectId})
            pdf_in.index = pdf_in['objectId_in']

            # Remove duplicates (keep the one with minimum distance)
            pdf_nodedup = pdf.loc[pdf.groupby('objectId').angDist.idxmin()]
            pdf_nodedup.index = pdf_nodedup['objectId']

            pdf_out = pdf_in.join(pdf_nodedup)

            # only for SIMBAD as we use `main_type` for our classification
            if 'main_type' in pdf_out.columns:
                pdf_out['main_type'] = pdf_out['main_type'].replace(np.nan, 'Unknown')

            if len(cols) > 1:
                # Concatenate all columns in one
                # use comma-separated values
                cols = [i.strip() for i in cols]
                pdf_out = pdf_out[cols]
                pdf_out['concat_cols'] = pdf_out.apply(lambda x: ','.join(x.astype(str).values.tolist()), axis=1)
                return pdf_out['concat_cols']
            elif len(cols) == 1:
                # single column to return
                return pdf_out[cols[0]].astype(str)

    except (ConnectionError, TimeoutError, ValueError) as ce:
        logging.warning("XMATCH failed " + repr(ce))
        ncols = len(cols.values[0].split(','))
        name = ','.join(["Fail"] * ncols)
        names = [name] * len(objectId)
        return pd.Series(names)

def xmatch_cds(
        df, catalogname='simbad', distmaxarcsec=1.0,
        cols_in=['candidate.candid', 'candidate.ra', 'candidate.dec'],
        cols_out=['main_type'],
        types=['string']):
    """ Cross-match Fink data from a Spark DataFrame with a catalog in CDS

    Parameters
    ----------
    df: Spark DataFrame
        Spark Dataframe
    catalogname: str
        Name of the catalog in Vizier, or directly simbad (default).
        Default is simbad.
    distmaxarcsec: float
        Cross-match radius in arcsecond. Default is 1.0 arcsecond.
    cols_in: list of str
        Three column names from the input DataFrame to use (oid, ra, dec).
        Default is [`candidate.candid`, `candidate.ra`, `candidate.dec`]
    cols_out: list of str
        N column names to get from the external catalog.
    types: list of str
        N types of columns from the external catalog.
        Should be SQL syntax (str=string, etc.)

    Returns
    ---------
    df_out: Spark DataFrame
        Spark DataFrame with new columns from the xmatch added

    Examples
    ---------
    >>> df = spark.read.load(ztf_alert_sample)

    # Simbad
    >>> df_simbad = xmatch_cds(df)
    >>> 'cdsxmatch' in df_simbad.columns
    True

    # Gaia
    >>> df_gaia = xmatch_cds(
    ...     df,
    ...     distmaxarcsec=1,
    ...     catalogname='vizier:I/355/gaiadr3',
    ...     cols_out=['DR3Name', 'Plx', 'e_Plx'],
    ...     types=['string', 'float', 'float'])
    >>> 'Plx' in df_gaia.columns
    True
    """
    df_out = df.withColumn(
        'xmatch',
        cdsxmatch(
            df[cols_in[0]],
            df[cols_in[1]],
            df[cols_in[2]],
            F.lit(distmaxarcsec),
            F.lit(catalogname),
            F.lit(','.join(cols_out))
        )
    ).withColumn('xmatch_split', F.split('xmatch', ','))

    for index, col_, type_ in zip(range(len(cols_out)), cols_out, types):
        df_out = df_out.withColumn(
            col_,
            F.col('xmatch_split').getItem(index).astype(type_)
        )

    df_out = df_out.drop('xmatch', 'xmatch_split')

    # Keep compatibility with previous definitions
    if 'main_type' in df_out.columns:
        # remove previous declaration if any
        df_out = df_out.drop('cdsxmatch')
        df_out = df_out.withColumnRenamed('main_type', 'cdsxmatch')

    return df_out


@pandas_udf(StringType(), PandasUDFType.SCALAR)
def crossmatch_other_catalog(candid, ra, dec, catalog_name, radius_arcsec=None):
    """ Crossmatch alerts with user-defined catalogs

    Currently supporting:
    - GCVS
    - VSX
    - 3HSP
    - 4LAC

    Parameters
    ----------
    candid: long
        ZTF candidate ID
    ra: float
        ZTF Right ascension
    dec: float
        ZTF declinations
    catalog_name: str
        Name of the catalog to use. currently supported: gcvs, vsx, 3hsp, 4lac
    radius_arcsec: float, optional
        Crossmatch radius in arcsecond. Default is 1.5 arcseconds.

    Returns
    ----------
    type: str
        Object type from the catalog. `Unknown` if no match.

    Examples
    ----------
    >>> from pyspark.sql.functions import lit

    Simulate fake data
    >>> ra = [26.8566983, 101.3520545, 0.3126, 0.31820833]
    >>> dec = [-26.9677112, 24.5421872, 47.6859, 29.59277778]
    >>> id = ["1", "2", "3", "4"]

    Wrap data into a Spark DataFrame
    >>> rdd = spark.sparkContext.parallelize(zip(id, ra, dec))
    >>> df = rdd.toDF(['id', 'ra', 'dec'])
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+
    | id|         ra|        dec|
    +---+-----------+-----------+
    |  1| 26.8566983|-26.9677112|
    |  2|101.3520545| 24.5421872|
    |  3|     0.3126|    47.6859|
    |  4| 0.31820833|29.59277778|
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
    |  3|     0.3126|    47.6859|Unknown|
    |  4| 0.31820833|29.59277778|Unknown|
    +---+-----------+-----------+-------+
    <BLANKLINE>

    >>> df.withColumn(
    ...     'vsx',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('vsx'))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+-------+
    | id|         ra|        dec|    vsx|
    +---+-----------+-----------+-------+
    |  1| 26.8566983|-26.9677112|   MISC|
    |  2|101.3520545| 24.5421872|   RRAB|
    |  3|     0.3126|    47.6859|Unknown|
    |  4| 0.31820833|29.59277778|Unknown|
    +---+-----------+-----------+-------+
    <BLANKLINE>

    >>> df.withColumn(
    ...     '3hsp',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('3hsp'))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+--------------------+
    | id|         ra|        dec|                3hsp|
    +---+-----------+-----------+--------------------+
    |  1| 26.8566983|-26.9677112|             Unknown|
    |  2|101.3520545| 24.5421872|             Unknown|
    |  3|     0.3126|    47.6859|             Unknown|
    |  4| 0.31820833|29.59277778|3HSPJ000116.4+293534|
    +---+-----------+-----------+--------------------+
    <BLANKLINE>

    >>> df.withColumn(
    ...     '4lac',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('4lac'), lit(60.0))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+-----------------+
    | id|         ra|        dec|             4lac|
    +---+-----------+-----------+-----------------+
    |  1| 26.8566983|-26.9677112|          Unknown|
    |  2|101.3520545| 24.5421872|          Unknown|
    |  3|     0.3126|    47.6859|4FGL J0001.2+4741|
    |  4| 0.31820833|29.59277778|          Unknown|
    +---+-----------+-----------+-----------------+
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
    elif catalog_name.values[0] == '3hsp':
        catalog = curdir + '/data/catalogs/3hsp.csv'
        ra2, dec2, type2 = extract_3hsp(catalog)
    elif catalog_name.values[0] == '4lac':
        catalog_h = curdir + '/data/catalogs/table-4LAC-DR3-h.fits'
        catalog_l = curdir + '/data/catalogs/table-4LAC-DR3-l.fits'
        ra2, dec2, type2 = extract_4lac(catalog_h, catalog_l)

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.values, dtype=float) * u.degree,
        dec=np.array(dec.values, dtype=float) * u.degree
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.values, dtype=float) * u.degree,
        dec=np.array(dec2.values, dtype=float) * u.degree
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_other, radius_arcsec=radius_arcsec
    )

    pdf_merge['Type'] = 'Unknown'
    pdf_merge.loc[mask, 'Type'] = [
        str(i).strip() for i in type2.astype(str).values[idx2]
    ]

    return pdf_merge['Type']

@pandas_udf(MapType(StringType(), StringType()), PandasUDFType.SCALAR)
def crossmatch_mangrove(candid, ra, dec, radius_arcsec=None):
    """ Crossmatch alerts with the Mangrove catalog

    Parameters
    ----------
    candid: long
        ZTF candidate ID
    ra: float
        ZTF Right ascension
    dec: float
        ZTF declinations
    radius_arcsec: float, optional
        Crossmatch radius in arcsecond. Default is 1.5 arcseconds.

    Returns
    ----------
    type: str
        Object type from the catalog. `Unknown` if no match.

    Examples
    ----------
    >>> from pyspark.sql.functions import lit

    Simulate fake data
    >>> ra = [198.955536, 101.3520545, 0.3126, 0.31820833]
    >>> dec = [42.029289, 24.5421872, 47.6859, 29.59277778]
    >>> id = ["1", "2", "3", "4"]

    Wrap data into a Spark DataFrame
    >>> rdd = spark.sparkContext.parallelize(zip(id, ra, dec))
    >>> df = rdd.toDF(['id', 'ra', 'dec'])
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+
    | id|         ra|        dec|
    +---+-----------+-----------+
    |  1| 198.955536|  42.029289|
    |  2|101.3520545| 24.5421872|
    |  3|     0.3126|    47.6859|
    |  4| 0.31820833|29.59277778|
    +---+-----------+-----------+
    <BLANKLINE>

    Test the processor by adding a new column with the result of the xmatch
    >>> df.withColumn(
    ...     'mangrove',
    ...     crossmatch_mangrove(df['id'], df['ra'], df['dec'], lit(60.0))
    ... ).toPandas() # doctest: +NORMALIZE_WHITESPACE
      id          ra        dec                                           mangrove
    0  1  198.955536  42.029289  {'HyperLEDA_name': 'NGC5055', '2MASS_name': '1...
    1  2  101.352054  24.542187  {'HyperLEDA_name': 'None', '2MASS_name': 'None...
    2  3    0.312600  47.685900  {'HyperLEDA_name': 'None', '2MASS_name': 'None...
    3  4    0.318208  29.592778  {'HyperLEDA_name': 'None', '2MASS_name': 'None...
    """
    pdf = pd.DataFrame(
        {
            'ra': ra.values,
            'dec': dec.values,
            'candid': range(len(ra))
        }
    )

    curdir = os.path.dirname(os.path.abspath(__file__))
    catalog = curdir + '/data/catalogs/mangrove_filtered.parquet'
    ra2, dec2, payload = extract_mangrove(catalog)

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.values, dtype=float) * u.degree,
        dec=np.array(dec.values, dtype=float) * u.degree
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.values, dtype=float) * u.degree,
        dec=np.array(dec2.values, dtype=float) * u.degree
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_other, radius_arcsec=radius_arcsec
    )

    default = {name: 'None' for name in MANGROVE_COLS}
    pdf_merge['Type'] = [default for i in range(len(pdf_merge))]
    pdf_merge.loc[mask, 'Type'] = [
        i for i in np.array(payload)[idx2]
    ]

    return pdf_merge['Type']


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
