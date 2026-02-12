# Copyright 2019-2025 AstroLab Software
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

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType, MapType, StructType, StructField

from astropy.coordinates import SkyCoord
from astropy import units as u

import io
import os
import logging
import requests
import pandas as pd
import numpy as np

from fink_science.rubin.xmatch.utils import cross_match_astropy
from fink_science.rubin.xmatch.utils import generate_csv
from fink_science.rubin.xmatch.utils import extract_vsx, extract_gcvs
from fink_science.rubin.xmatch.utils import extract_3hsp, extract_4lac
from fink_science.rubin.xmatch.utils import extract_mangrove, MANGROVE_COLS
from fink_science.rubin.xmatch.utils import (
    extract_tns,
    TNS_COLS,
    TNS_TYPES,
    TNS_SPARK_SCHEMA,
)
from fink_science.tester import spark_unit_tests
from fink_science import __file__

from fink_tns.utils import download_catalog

from typing import Any

_LOG = logging.getLogger(__name__)


@pandas_udf(StringType(), PandasUDFType.SCALAR)
@profile
def cdsxmatch(
    diaSourceId: Any,
    ra: Any,
    dec: Any,
    distmaxarcsec: float,
    extcatalog: str,
    cols: str,
) -> pd.Series:
    """Query the CDSXmatch service to find identified objects in alerts.

    Notes
    -----
    The catalog queried is the SIMBAD bibliographical database.

    Notes
    -----
    Assuming 2 output fields, a returned row can have the following values:
    - "a,b"       --> match
    - None        --> No match
    - "Fail,Fail" --> error from the service

    Parameters
    ----------
    diaSourceId: list of str or Spark DataFrame Column of str
        List containing alert ids (custom)
    ra: list of float or Spark DataFrame Column of float
        List containing alert ra coordinates
    dec: list of float or Spark DataFrame Column of float
        List containing alert dec coordinates
    distmaxarcsec: list of float or Spark DataFrame Column of float
        Cross-match radius in arcsecond
    extcatalog: list of str or Spark DataFrame Column of str
        Name of the external catalog in Vizier, or directly simbad.
    cols: list of str or Spark DataFrame Column of str
        Comma-separated column names to get from the external catalog

    Returns
    -------
    out: pandas.Series of string
        Return Pandas DataFrame with a new single column
        containing comma-separated values of extra-columns.
        If the object is not found in Simbad, the type is
        marked as null. In the case several objects match
        the centroid of the alert, only the closest is returned.
        If the request Failed (no match at all), return Column of Fails.

    Examples
    --------
    Simulate fake data
    >>> ra = [26.8566983, 26.24497, 1.0]
    >>> dec = [-26.9677112, -26.7569436, 0.0]
    >>> id = ["a", "b", "c"]

    Wrap data into a Spark DataFrame
    >>> rdd = spark.sparkContext.parallelize(zip(id, ra, dec))
    >>> df = rdd.toDF(['id', 'ra', 'dec'])
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+----------+-----------+
    | id|        ra|        dec|
    +---+----------+-----------+
    |  a|26.8566983|-26.9677112|
    |  b|  26.24497|-26.7569436|
    |  c|       1.0|        0.0|
    +---+----------+-----------+
    <BLANKLINE>

    Test the processor by adding a new column with the result of the xmatch
    >>> df = df.withColumn(
    ...     'simbad_otype',
    ...     cdsxmatch(
    ...         df['id'], df['ra'], df['dec'],
    ...         F.lit(1.0), F.lit('simbad'), F.lit('otype')))
    >>> df.show() # doctest: +NORMALIZE_WHITESPACE
    +---+----------+-----------+------------+
    | id|        ra|        dec|simbad_otype|
    +---+----------+-----------+------------+
    |  a|26.8566983|-26.9677112|         LP*|
    |  b|  26.24497|-26.7569436|           *|
    |  c|       1.0|        0.0|        null|
    +---+----------+-----------+------------+
    <BLANKLINE>
    """
    # If nothing
    if len(ra) == 0:
        return pd.Series([])

    # Catch TimeoutError and ConnectionError
    try:
        # Build a catalog of alert in a CSV-like string
        table_header = """ra_in,dec_in,diaSourceId\n"""
        table = generate_csv(table_header, [ra, dec, diaSourceId])

        # Send the request!
        r = requests.post(
            "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
            data={
                "request": "xmatch",
                "distMaxArcsec": distmaxarcsec.to_numpy()[0],
                "selection": "all",
                "RESPONSEFORMAT": "csv",
                "cat2": extcatalog.to_numpy()[0],
                "cols2": cols.to_numpy()[0],
                "colRA1": "ra_in",
                "colDec1": "dec_in",
            },
            files={"cat1": table},
        )

        col_list = cols.to_numpy()[0].split(",")

        if r.status_code != 200:
            msg = "Fail"
            names = [",".join([msg] * len(col_list))] * len(diaSourceId)

            # Error, return "Fail,Fail,Fail..."
            return pd.Series(names)
        else:
            pdf = pd.read_csv(io.BytesIO(r.content))

            if pdf.empty:
                # null values
                name = None
                names = [name] * len(diaSourceId)
                # No error, but no match, return None (null values for Spark)
                return pd.Series(names)

            # join
            pdf_in = pd.DataFrame({"diaSourceId_in": diaSourceId})
            pdf_in.index = pdf_in["diaSourceId_in"]

            # Remove duplicates (keep the one with minimum distance)
            pdf_nodedup = pdf.loc[pdf.groupby("diaSourceId").angDist.idxmin()]
            pdf_nodedup.index = pdf_nodedup["diaSourceId"]

            pdf_out = pdf_in.join(pdf_nodedup)

            # To get null values in Spark
            pdf_out = pdf_out.replace(np.nan, None)

            if len(col_list) > 1:
                # Concatenate all columns in one
                # use comma-separated values
                col_list = [i.strip() for i in col_list]
                pdf_out = pdf_out[col_list]
                pdf_out["concat_cols"] = pdf_out.apply(
                    lambda row: (
                        None
                        if all(row[col] is None for col in col_list)
                        else ",".join(
                            str(row[col]) for col in col_list if row[col] is not None
                        )
                    ),
                    axis=1,
                )
                # No error, possible matches, several columns to return: return "a,b"
                return pdf_out["concat_cols"]
            elif len(col_list) == 1:
                # No error, possible matches, single column: return "a"
                return pdf_out[col_list[0]].apply(
                    lambda x: None if x is None else str(x)
                )

    except (ConnectionError, TimeoutError, ValueError) as ce:
        logging.warning("XMATCH failed " + repr(ce))
        name = ",".join(["Fail"] * len(col_list))
        names = [name] * len(diaSourceId)
        return pd.Series(names)


def xmatch_cds(
    df,
    catalogname="simbad",
    prefix_col_out=None,
    distmaxarcsec=1.0,
    cols_in=None,
    cols_out=None,
    types=None,
):
    """Cross-match Fink data from a Spark DataFrame with a catalog in CDS

    Notes
    -----
    To check available columns and their name:
    http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync/tables?action=getColList&tabName=I/355/gaiadr3&RESPONSEFORMAT=json

    Parameters
    ----------
    df: Spark DataFrame
        Spark Dataframe
    catalogname: str
        Name of the catalog in Vizier, or directly simbad (default).
        Default is simbad.
    prefix_col_out: str
        Output DataFrame columns will be named
        <prefix_col_out>_<col>. If None,
        `prefix_col_out`=`catalogname`, but set it if there are
        illegal characters in the catalogname.

    distmaxarcsec: float
        Cross-match radius in arcsecond. Default is 1.0 arcsecond.
    cols_in: list of str
        Three column names from the input DataFrame to use (oid, ra, dec).
        Default is ["diaSource.diaSourceId", "diaSource.ra", "diaSource.dec"]
    cols_out: list of str
        N column names to get from the external catalog.
        If None, assume ["otype"] for simbad.
    types: list of str
        N types of columns from the external catalog.
        Should be SQL syntax (str=string, etc.).
        If None, return ["str"] for simbad.

    Returns
    -------
    df_out: Spark DataFrame
        Spark DataFrame with new columns from the xmatch added

    Examples
    --------
    >>> df = spark.read.load(rubin_alert_sample)

    # Simbad
    >>> df_simbad = xmatch_cds(df)
    >>> 'simbad_otype' in df_simbad.columns
    True

    # Gaia
    >>> df_gaia = xmatch_cds(
    ...     df,
    ...     distmaxarcsec=1,
    ...     catalogname='vizier:I/355/gaiadr3',
    ...     prefix_col_out="gaiadr3",
    ...     cols_out=['DR3Name', 'Plx', 'e_Plx'],
    ...     types=['string', 'float', 'float'])
    >>> 'gaiadr3_Plx' in df_gaia.columns
    True

    # VSX
    >>> df_vsx = xmatch_cds(
    ...     df,
    ...     catalogname="vizier:B/vsx/vsx",
    ...     distmaxarcsec=1.5,
    ...     cols_out=['Type'],
    ...     types=['string'])
    >>> 'vizier:B/vsx/vsx_Type' in df_vsx.columns
    True

    # SPICY
    >>> df_spicy = xmatch_cds(
    ...     df,
    ...     catalogname="vizier:J/ApJS/254/33/table1",
    ...     distmaxarcsec=1.2,
    ...     cols_out=['SPICY', 'class'],
    ...     types=['int', 'string'])
    >>> 'vizier:J/ApJS/254/33/table1_SPICY' in df_spicy.columns
    True
    """
    if cols_in is None:
        cols_in = ["diaSource.diaSourceId", "diaSource.ra", "diaSource.dec"]
    if cols_out is None:
        cols_out = ["otype"]
    if types is None:
        types = ["string"]

    if prefix_col_out is None:
        prefix_col_out = catalogname

    df_out = df.withColumn(
        "xmatch",
        cdsxmatch(
            df[cols_in[0]],
            df[cols_in[1]],
            df[cols_in[2]],
            F.lit(distmaxarcsec),
            F.lit(catalogname),
            F.lit(",".join(cols_out)),
        ),
    )

    for index, col_, type_ in zip(range(len(cols_out)), cols_out, types):
        df_out = df_out.withColumn(
            "{}_{}".format(prefix_col_out, col_),
            F.split("xmatch", ",").getItem(index).astype(type_),
        )

    df_out = df_out.drop("xmatch")

    return df_out


def xmatch_tns(df, distmaxarcsec=1.5, tns_raw_output=""):
    """Cross-match Fink data from a Spark DataFrame with the latest TNS catalog

    Notes
    -----
    From 2026/01, we crossmatch with all entries - not only the confirmed ones.

    Parameters
    ----------
    df: Spark DataFrame
        Spark Dataframe
    distmaxarcsec: float, optional
        Cross-match radius in arcsecond. Default is 1.5 arcsecond.
    tns_raw_output: str, optional
        Folder that contains raw TNS catalog. Inside, it is expected
        to find the file `tns_raw.parquet` downloaded using
        `fink-broker/bin/download_tns.py`. Default is None, in
        which case the catalog will be downloaded. Beware that
        to download the catalog, you need to set environment variables:
        - TNS_API_MARKER: path to the TNS API marker (tns_marker.txt)
        - TNS_API_KEY: path to the TNS API key (tns_api.key)

    Returns
    -------
    df: Spark DataFrame
        Spark DataFrame with new columns from the xmatch added

    Examples
    --------
    >>> df = spark.read.load(rubin_alert_sample)

    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> path = curdir + '/data/catalogs'
    >>> df_tns = xmatch_tns(df, tns_raw_output=path)
    >>> assert 'tns_type' in df_tns.columns, df_tns.columns
    >>> assert 'tns_redshift' in df_tns.columns, df_tns.columns

    >>> df_tns.filter(df_tns["tns_type"].isNull()).count()
    100

    """
    if tns_raw_output == "":
        _LOG.info("Downloading the latest TNS catalog")
        if "TNS_API_MARKER" in os.environ and "TNS_API_KEY" in os.environ:
            with open(os.environ["TNS_API_MARKER"]) as f:
                tns_marker = f.read().replace("\n", "")

            pdf_tns = download_catalog(os.environ["TNS_API_KEY"], tns_marker)
        else:
            _LOG.warning(
                "TNS_API_MARKER and TNS_API_KEY are not defined as env var in the master."
            )
            _LOG.warning(
                "Skipping crossmatch with TNS. Creating columns with null values."
            )
            for col_name, col_type in zip(TNS_COLS, TNS_TYPES):
                df = df.withColumn(
                    "tns_{}".format(col_name), F.lit(None).astype(col_type)
                )
            return df
    else:
        pdf_tns = pd.read_parquet(os.path.join(tns_raw_output, "tns_raw.parquet"))

    spark = SparkSession.builder.getOrCreate()
    pdf_tns_b = spark.sparkContext.broadcast(pdf_tns)

    tns_schema = StructType([
        StructField(k, v, True) for k, v in TNS_SPARK_SCHEMA.items()
    ])

    @pandas_udf(tns_schema, PandasUDFType.SCALAR)
    def crossmatch_with_tns(diaSourceId, ra, dec):
        """Spark pandas_udf to crossmatch Rubin alerts with TNS

        Parameters
        ----------
        diaSourceId: pd.Series of str
            Alert diaSourceId
        ra: pd.Series of double
            Alert RA position
        dec: pd.Series of double
            Alert Dec position

        Returns
        -------
        to_return: pd.Series of dict
            TNS name and type for the alert. null if no match.
        """
        pdf_lsst = pd.DataFrame({
            "diaSourceId": range(len(ra)),
            "ra": ra.to_numpy(),
            "dec": dec.to_numpy(),
        })

        ra2, dec2, payload = extract_tns(pdf_tns_b.value)

        # create catalogs
        catalog_lsst = SkyCoord(
            ra=np.array(ra, dtype=float) * u.degree,
            dec=np.array(dec, dtype=float) * u.degree,
        )
        catalog_tns = SkyCoord(
            ra=np.array(ra2, dtype=float) * u.degree,
            dec=np.array(dec2, dtype=float) * u.degree,
        )

        pdf_merge, mask, idx2 = cross_match_astropy(
            pdf_lsst, catalog_lsst, catalog_tns, radius_arcsec=distmaxarcsec
        )

        default = [None] * len(TNS_SPARK_SCHEMA)
        pdf_merge["return"] = pd.Series([default for i in range(len(pdf_merge))])
        pdf_merge.loc[mask, "return"] = pd.Series([
            [None if pd.isna(x) else x for x in payload[i].tolist()] for i in idx2
        ]).values

        out = pd.DataFrame.from_dict(
            dict(zip(pdf_merge["return"].index, pdf_merge["return"].values)),
            columns=TNS_COLS,
            orient="index",
        )
        return out

    df = df.withColumn(
        "tns",
        crossmatch_with_tns(
            df["diaSource.diaSourceId"], df["diaSource.ra"], df["diaSource.dec"]
        ),
    )

    # Explode TNS
    for col_ in TNS_COLS:
        df = df.withColumn(
            "tns_{}".format(col_),
            df["tns"].getItem(col_),
        )
    df = df.drop("tns")

    return df


@pandas_udf(StringType(), PandasUDFType.SCALAR)
@profile
def crossmatch_other_catalog(diaSourceId, ra, dec, catalog_name, radius_arcsec=None):
    """Crossmatch alerts with user-defined catalogs

    Currently supporting:
    - GCVS
    - VSX
    - 3HSP
    - 4LAC

    Parameters
    ----------
    diaSourceId: long
        Rubin diaSourceId
    ra: float
        Rubin Right ascension
    dec: float
        Rubin declinations
    catalog_name: str
        Name of the catalog to use. currently supported: gcvs, vsx, 3hsp, 4lac
    radius_arcsec: float, optional
        Crossmatch radius in arcsecond. Default is 1.5 arcseconds.

    Returns
    -------
    type: str
        Object type from the catalog. null if no match.

    Examples
    --------
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
    +---+-----------+-----------+----+
    | id|         ra|        dec|gcvs|
    +---+-----------+-----------+----+
    |  1| 26.8566983|-26.9677112|null|
    |  2|101.3520545| 24.5421872|  RR|
    |  3|     0.3126|    47.6859|null|
    |  4| 0.31820833|29.59277778|null|
    +---+-----------+-----------+----+
    <BLANKLINE>

    >>> df.withColumn(
    ...     'vsx',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('vsx'))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+----+
    | id|         ra|        dec| vsx|
    +---+-----------+-----------+----+
    |  1| 26.8566983|-26.9677112|MISC|
    |  2|101.3520545| 24.5421872|RRAB|
    |  3|     0.3126|    47.6859|null|
    |  4| 0.31820833|29.59277778|null|
    +---+-----------+-----------+----+
    <BLANKLINE>

    >>> df.withColumn(
    ...     '3hsp',
    ...     crossmatch_other_catalog(df['id'], df['ra'], df['dec'], lit('3hsp'))
    ... ).show() # doctest: +NORMALIZE_WHITESPACE
    +---+-----------+-----------+--------------------+
    | id|         ra|        dec|                3hsp|
    +---+-----------+-----------+--------------------+
    |  1| 26.8566983|-26.9677112|                null|
    |  2|101.3520545| 24.5421872|                null|
    |  3|     0.3126|    47.6859|                null|
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
    |  1| 26.8566983|-26.9677112|             null|
    |  2|101.3520545| 24.5421872|             null|
    |  3|     0.3126|    47.6859|4FGL J0001.2+4741|
    |  4| 0.31820833|29.59277778|             null|
    +---+-----------+-----------+-----------------+
    <BLANKLINE>
    """
    pdf = pd.DataFrame({
        "ra": ra.to_numpy(),
        "dec": dec.to_numpy(),
        "diaSourceId": range(len(ra)),
    })

    curdir = os.path.dirname(os.path.abspath(__file__))
    if catalog_name.to_numpy()[0] == "gcvs":
        catalog = curdir + "/data/catalogs/gcvs.parquet"
        ra2, dec2, type2 = extract_gcvs(catalog)
    elif catalog_name.to_numpy()[0] == "vsx":
        catalog = curdir + "/data/catalogs/vsx.parquet"
        ra2, dec2, type2 = extract_vsx(catalog)
    elif catalog_name.to_numpy()[0] == "3hsp":
        catalog = curdir + "/data/catalogs/3hsp.csv"
        ra2, dec2, type2 = extract_3hsp(catalog)
    elif catalog_name.to_numpy()[0] == "4lac":
        catalog_h = curdir + "/data/catalogs/table-4LAC-DR3-h.fits"
        catalog_l = curdir + "/data/catalogs/table-4LAC-DR3-l.fits"
        ra2, dec2, type2 = extract_4lac(catalog_h, catalog_l)

    # create catalogs
    catalog_rubin = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec2.to_numpy(), dtype=float) * u.degree,
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_rubin, catalog_other, radius_arcsec=radius_arcsec
    )

    pdf_merge["Type"] = None
    pdf_merge.loc[mask, "Type"] = [
        str(i).strip() for i in type2.astype(str).to_numpy()[idx2]
    ]

    return pdf_merge["Type"]


@pandas_udf(MapType(StringType(), StringType()), PandasUDFType.SCALAR)
@profile
def crossmatch_mangrove(diaSourceId, ra, dec, radius_arcsec=None):
    """Crossmatch alerts with the Mangrove catalog

    Parameters
    ----------
    diaSourceId: long
        Rubin diaSourceId
    ra: float
        Rubin Right ascension
    dec: float
        Rubin declinations
    radius_arcsec: float, optional
        Crossmatch radius in arcsecond. Default is 1.5 arcseconds.

    Returns
    -------
    type: str
        Object type from the catalog. null if no match.

    Examples
    --------
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
    ... ).select("mangrove.HyperLEDA_name").show()
    +--------------+
    |HyperLEDA_name|
    +--------------+
    |       NGC5055|
    |          null|
    |          null|
    |          null|
    +--------------+
    <BLANKLINE>
    """
    pdf = pd.DataFrame({
        "ra": ra.to_numpy(),
        "dec": dec.to_numpy(),
        "diaSourceId": range(len(ra)),
    })

    curdir = os.path.dirname(os.path.abspath(__file__))
    catalog = curdir + "/data/catalogs/mangrove_filtered.parquet"
    ra2, dec2, payload = extract_mangrove(catalog)

    # create catalogs
    catalog_rubin = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec2.to_numpy(), dtype=float) * u.degree,
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_rubin, catalog_other, radius_arcsec=radius_arcsec
    )

    default = {name: None for name in MANGROVE_COLS}
    pdf_merge["Type"] = [default for i in range(len(pdf_merge))]
    pdf_merge.loc[mask, "Type"] = [payload[i] for i in idx2]

    return pdf_merge["Type"]


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    # from fink-alerts-schemas (see CI configuration)
    rubin_alert_sample = "file://{}/datasim/rubin_test_data_10_0.parquet".format(path)
    globs["rubin_alert_sample"] = rubin_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
