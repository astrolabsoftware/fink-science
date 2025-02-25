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
from pyspark.sql.types import StringType, MapType

from astropy.coordinates import SkyCoord
from astropy import units as u

import io
import os
import logging
import requests
import pandas as pd
import numpy as np

from fink_science.ztf.xmatch.utils import cross_match_astropy
from fink_science.ztf.xmatch.utils import generate_csv
from fink_science.ztf.xmatch.utils import extract_vsx, extract_gcvs
from fink_science.ztf.xmatch.utils import extract_3hsp, extract_4lac
from fink_science.ztf.xmatch.utils import extract_mangrove, MANGROVE_COLS
from fink_science.tester import spark_unit_tests
from fink_science import __file__

from fink_tns.utils import download_catalog

from typing import Any

_LOG = logging.getLogger(__name__)


@pandas_udf(StringType(), PandasUDFType.SCALAR)
@profile
def cdsxmatch(
    objectId: Any, ra: Any, dec: Any, distmaxarcsec: float, extcatalog: str, cols: str
) -> pd.Series:
    """Query the CDSXmatch service to find identified objects in alerts.

    The catalog queried is the SIMBAD bibliographical database.

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
    -------
    out: pandas.Series of string
        Return Pandas DataFrame with a new single column
        containing comma-separated values of extra-columns.
        If the object is not found in Simbad, the type is
        marked as Unknown. In the case several objects match
        the centroid of the alert, only the closest is returned.
        If the request Failed (no match at all), return Column of Fail.

    Examples
    --------
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
    +---+----------+-----------+------------+
    | id|        ra|        dec|   cdsxmatch|
    +---+----------+-----------+------------+
    |  a|26.8566983|-26.9677112|LongPeriodV*|
    |  b|  26.24497|-26.7569436|        Star|
    +---+----------+-----------+------------+
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

        if r.status_code != 200:
            names = ["Fail {}".format(r.status_code)] * len(objectId)
            return pd.Series(names)
        else:
            cols = cols.to_numpy()[0].split(",")
            pdf = pd.read_csv(io.BytesIO(r.content))

            if pdf.empty:
                name = ",".join(["Unknown"] * len(cols))
                names = [name] * len(objectId)
                return pd.Series(names)

            # join
            pdf_in = pd.DataFrame({"objectId_in": objectId})
            pdf_in.index = pdf_in["objectId_in"]

            # Remove duplicates (keep the one with minimum distance)
            pdf_nodedup = pdf.loc[pdf.groupby("objectId").angDist.idxmin()]
            pdf_nodedup.index = pdf_nodedup["objectId"]

            pdf_out = pdf_in.join(pdf_nodedup)

            # only for SIMBAD as we use `main_type` for our classification
            if "main_type" in pdf_out.columns:
                pdf_out["main_type"] = pdf_out["main_type"].replace(np.nan, "Unknown")

            if len(cols) > 1:
                # Concatenate all columns in one
                # use comma-separated values
                cols = [i.strip() for i in cols]
                pdf_out = pdf_out[cols]
                pdf_out["concat_cols"] = pdf_out.apply(
                    lambda x: ",".join(x.astype(str).to_numpy().tolist()), axis=1
                )
                return pdf_out["concat_cols"]
            elif len(cols) == 1:
                # single column to return
                return pdf_out[cols[0]].astype(str)

    except (ConnectionError, TimeoutError, ValueError) as ce:
        logging.warning("XMATCH failed " + repr(ce))
        ncols = len(cols.to_numpy()[0].split(","))
        name = ",".join(["Fail"] * ncols)
        names = [name] * len(objectId)
        return pd.Series(names)


def xmatch_cds(
    df,
    catalogname="simbad",
    distmaxarcsec=1.0,
    cols_in=None,
    cols_out=None,
    types=None,
):
    """Cross-match Fink data from a Spark DataFrame with a catalog in CDS

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
    -------
    df_out: Spark DataFrame
        Spark DataFrame with new columns from the xmatch added

    Examples
    --------
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

    # VSX
    >>> df_vsx = xmatch_cds(
    ...     df,
    ...     catalogname="vizier:B/vsx/vsx",
    ...     distmaxarcsec=1.5,
    ...     cols_out=['Type'],
    ...     types=['string'])
    >>> 'Type' in df_vsx.columns
    True

    # SPICY
    >>> df_spicy = xmatch_cds(
    ...     df,
    ...     catalogname="vizier:J/ApJS/254/33/table1",
    ...     distmaxarcsec=1.2,
    ...     cols_out=['SPICY', 'class'],
    ...     types=['int', 'string'])
    >>> 'SPICY' in df_spicy.columns
    True
    """
    if cols_in is None:
        cols_in = ["candidate.candid", "candidate.ra", "candidate.dec"]
    if cols_out is None:
        cols_out = ["main_type"]
    if types is None:
        types = ["string"]

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
    ).withColumn("xmatch_split", F.split("xmatch", ","))

    for index, col_, type_ in zip(range(len(cols_out)), cols_out, types):
        df_out = df_out.withColumn(
            col_, F.col("xmatch_split").getItem(index).astype(type_)
        )

    df_out = df_out.drop("xmatch", "xmatch_split")

    # Keep compatibility with previous definitions
    if "main_type" in df_out.columns:
        # remove previous declaration if any
        df_out = df_out.drop("cdsxmatch")
        df_out = df_out.withColumnRenamed("main_type", "cdsxmatch")

    return df_out


def xmatch_tns(df, distmaxarcsec=1.5, tns_raw_output=""):
    """Cross-match Fink data from a Spark DataFrame with the latest TNS catalog

    Parameters
    ----------
    df: Spark DataFrame
        Spark Dataframe
    distmaxarcsec: float, optional
        Cross-match radius in arcsecond. Default is 1.5 arcsecond.
    tns_raw_output: str, optional
        Folder that contains raw TNS catalog. Inside, it is expected
        to find the file `tns_raw.parquet` downloaded using
        `fink-broker/bin/download_tns.py`. Default is "", in
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
    >>> df = spark.read.load(ztf_alert_sample)

    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> path = curdir + '/data/catalogs'
    >>> df_tns = xmatch_tns(df, tns_raw_output=path)
    >>> 'tns' in df_tns.columns
    True

    >>> df_tns.filter(df_tns["tns"] != "").count()
    1

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
            _LOG.warning("Skipping crossmatch with TNS.")
            df = df.withColumn("tns", F.lit(""))
            return df
    else:
        pdf_tns = pd.read_parquet(os.path.join(tns_raw_output, "tns_raw.parquet"))

    # Filter TNS confirmed data
    f1 = ~pdf_tns["type"].isna()
    pdf_tns_filt = pdf_tns[f1]

    spark = SparkSession.builder.getOrCreate()
    pdf_tns_filt_b = spark.sparkContext.broadcast(pdf_tns_filt)

    @pandas_udf(StringType(), PandasUDFType.SCALAR)
    def crossmatch_with_tns(objectid, ra, dec):
        """Spark pandas_udf to crossmatch ZTF alerts with TNS

        Parameters
        ----------
        objectid: pd.Series of str
            Alert objectId
        ra: pd.Series of double
            Alert RA position
        dec: pd.Series of double
            Alert Dec position

        Returns
        -------
        to_return: pd.Series of str
            TNS type for the alert. `Unknown` if no match.
        """
        pdf = pdf_tns_filt_b.value
        ra2, dec2, type2 = pdf["ra"], pdf["declination"], pdf["type"]

        # create catalogs
        catalog_ztf = SkyCoord(
            ra=np.array(ra, dtype=float) * u.degree,
            dec=np.array(dec, dtype=float) * u.degree,
        )
        catalog_tns = SkyCoord(
            ra=np.array(ra2, dtype=float) * u.degree,
            dec=np.array(dec2, dtype=float) * u.degree,
        )

        # cross-match
        _, _, _ = catalog_tns.match_to_catalog_sky(catalog_ztf)

        sub_pdf = pd.DataFrame({
            "objectId": objectid.to_numpy(),
            "ra": ra.to_numpy(),
            "dec": dec.to_numpy(),
        })

        # cross-match
        idx2, d2d2, _ = catalog_ztf.match_to_catalog_sky(catalog_tns)

        # set separation length
        sep_constraint2 = d2d2.degree < distmaxarcsec / 3600.0

        sub_pdf["TNS"] = [""] * len(sub_pdf)
        sub_pdf["TNS"][sep_constraint2] = type2.to_numpy()[idx2[sep_constraint2]]

        # Here we take the first match
        # What if there are many? AT & SN?
        to_return = objectid.apply(
            lambda x: ""
            if x not in sub_pdf["objectId"].to_numpy()
            else sub_pdf["TNS"][sub_pdf["objectId"] == x].to_numpy()[0]
        )

        return to_return

    df = df.withColumn(
        "tns",
        crossmatch_with_tns(df["objectId"], df["candidate.ra"], df["candidate.dec"]),
    )

    return df


@pandas_udf(StringType(), PandasUDFType.SCALAR)
@profile
def crossmatch_other_catalog(candid, ra, dec, catalog_name, radius_arcsec=None):
    """Crossmatch alerts with user-defined catalogs

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
    -------
    type: str
        Object type from the catalog. `Unknown` if no match.

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
    pdf = pd.DataFrame({
        "ra": ra.to_numpy(),
        "dec": dec.to_numpy(),
        "candid": range(len(ra)),
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
    catalog_ztf = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec2.to_numpy(), dtype=float) * u.degree,
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_other, radius_arcsec=radius_arcsec
    )

    pdf_merge["Type"] = "Unknown"
    pdf_merge.loc[mask, "Type"] = [
        str(i).strip() for i in type2.astype(str).to_numpy()[idx2]
    ]

    return pdf_merge["Type"]


@pandas_udf(MapType(StringType(), StringType()), PandasUDFType.SCALAR)
@profile
def crossmatch_mangrove(candid, ra, dec, radius_arcsec=None):
    """Crossmatch alerts with the Mangrove catalog

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
    -------
    type: str
        Object type from the catalog. `Unknown` if no match.

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
    ... ).toPandas() # doctest: +NORMALIZE_WHITESPACE
      id          ra        dec                                           mangrove
    0  1  198.955536  42.029289  {'HyperLEDA_name': 'NGC5055', '2MASS_name': '1...
    1  2  101.352054  24.542187  {'HyperLEDA_name': 'None', '2MASS_name': 'None...
    2  3    0.312600  47.685900  {'HyperLEDA_name': 'None', '2MASS_name': 'None...
    3  4    0.318208  29.592778  {'HyperLEDA_name': 'None', '2MASS_name': 'None...
    """
    pdf = pd.DataFrame({
        "ra": ra.to_numpy(),
        "dec": dec.to_numpy(),
        "candid": range(len(ra)),
    })

    curdir = os.path.dirname(os.path.abspath(__file__))
    catalog = curdir + "/data/catalogs/mangrove_filtered.parquet"
    ra2, dec2, payload = extract_mangrove(catalog)

    # create catalogs
    catalog_ztf = SkyCoord(
        ra=np.array(ra.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec.to_numpy(), dtype=float) * u.degree,
    )

    catalog_other = SkyCoord(
        ra=np.array(ra2.to_numpy(), dtype=float) * u.degree,
        dec=np.array(dec2.to_numpy(), dtype=float) * u.degree,
    )

    pdf_merge, mask, idx2 = cross_match_astropy(
        pdf, catalog_ztf, catalog_other, radius_arcsec=radius_arcsec
    )

    default = {name: "None" for name in MANGROVE_COLS}
    pdf_merge["Type"] = [default for i in range(len(pdf_merge))]
    pdf_merge.loc[mask, "Type"] = [payload[i] for i in idx2]

    return pdf_merge["Type"]


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
