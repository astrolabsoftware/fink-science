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

import io
import csv
import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from fink_science.tester import regular_unit_tests

MANGROVE_COLS = ["HyperLEDA_name", "2MASS_name", "lum_dist", "ang_dist"]


@profile
def cross_match_astropy(pdf, catalog_ztf, catalog_other, radius_arcsec=None):
    """Crossmatch two catalogs

    Parameters
    ----------
    pdf: pd.DataFrame
        Input alert dataframe
    catalog_ztf: SkyCoord
        Catalog of ZTF objects
    catalog_other: SkyCoord
        Catalog of other objects
    radius_arcsec: pd.Series or None
        If None, default is 1.5 arcsec
    """
    # cross-match
    idx, d2d, _ = catalog_other.match_to_catalog_sky(catalog_ztf)

    # set separation length
    if radius_arcsec is None:
        radius_arcsec = 1.5
    else:
        radius_arcsec = float(radius_arcsec.to_numpy()[0])

    sep_constraint = d2d.degree < radius_arcsec / 3600.0

    catalog_matches = np.unique(pdf["diaSourceId"].to_numpy()[idx[sep_constraint]])

    # identify position of matches in the input dataframe
    pdf_matches = pd.DataFrame({
        "diaSourceId": np.array(catalog_matches, dtype=np.int64),
        "match": True,
    })
    pdf_merge = pdf.merge(pdf_matches, how="left", on="diaSourceId")

    mask = pdf_merge["match"].apply(lambda x: x is True)

    # Now get types for these
    catalog_ztf_merge = SkyCoord(
        ra=np.array(pdf_merge.loc[mask, "ra"].to_numpy(), dtype=float) * u.degree,
        dec=np.array(pdf_merge.loc[mask, "dec"].to_numpy(), dtype=float) * u.degree,
    )

    # cross-match
    idx2, _, _ = catalog_ztf_merge.match_to_catalog_sky(catalog_other)

    return pdf_merge, mask, idx2


def extract_mangrove(filename):
    """Read the Mangrove catalog and extract useful columns

    Parameters
    ----------
    filename: str
        Path to the Mangrove catalog (parquet file)

    Returns
    -------
    out: pd.Series, pd.Series, pd.Series
        (ra, dec, Source Name )from the catalog

    Examples
    --------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> filename = curdir + '/../../data/catalogs/mangrove_filtered.parquet'
    >>> ra, dec, payload = extract_mangrove(filename)
    """
    pdf = pd.read_parquet(filename)

    cols = ["HyperLEDA_name", "2MASS_name", "lum_dist", "ang_dist"]

    for col_ in cols:
        # stringify
        pdf[col_] = pdf[col_].astype(str)

    payload = pdf[cols].to_dict(orient="records")

    return pdf["ra"], pdf["dec"], payload


def extract_4lac(filename_h, filename_l):
    """Read the 4LAC DR3 catalogs and extract useful columns

    Parameters
    ----------
    filename{i}: str
        Path to the high and low latitudes catalogs (fits file)

    Returns
    -------
    out: pd.Series, pd.Series, pd.Series
        (ra, dec, Source Name )from the catalog

    Examples
    --------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> catalog_h = curdir + '/../../data/catalogs/table-4LAC-DR3-h.fits'
    >>> catalog_l = curdir + '/../../data/catalogs/table-4LAC-DR3-l.fits'
    >>> ra, dec, sourcename = extract_4lac(catalog_h, catalog_l)
    """
    dat = Table.read(filename_h, format="fits")
    pdf_4lac_h = dat.to_pandas()

    dat = Table.read(filename_l, format="fits")
    pdf_4lac_l = dat.to_pandas()

    pdf_4lac = pd.concat([pdf_4lac_h, pdf_4lac_l])

    # decode as str
    pdf_4lac["Source_Name"] = pdf_4lac["Source_Name"].apply(lambda x: x.decode())

    return pdf_4lac["RAJ2000"], pdf_4lac["DEJ2000"], pdf_4lac["Source_Name"]


def extract_3hsp(filename):
    """Read the 3HSP catalog and extract useful columns

    Parameters
    ----------
    filename: str
        Path to the catalog (csv file)

    Returns
    -------
    out: pd.Series, pd.Series, pd.Series
        (ra, dec, Source Name )from the catalog

    Examples
    --------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> catalog = curdir + '/../../data/catalogs/3hsp.csv'
    >>> ra, dec, sourcename = extract_3hsp(catalog)
    """
    pdf_3hsp = pd.read_csv(filename, header=0)

    # remove white spaces around column names
    pdf_3hsp = pdf_3hsp.rename(columns={i: i.strip() for i in pdf_3hsp.columns})

    # convert RA/Dec into degrees
    pdf_3hsp["R.A."] = pdf_3hsp["R.A."].apply(lambda x: x.strip().replace('"', ""))
    pdf_3hsp["Dec"] = pdf_3hsp["Dec"].apply(lambda x: x.strip().replace('"', ""))

    coord = SkyCoord(
        pdf_3hsp[["R.A.", "Dec"]]
        .apply(lambda x: "{} {}".format(*x), axis=1)
        .to_numpy(),
        unit=(u.hourangle, u.deg),
    )

    pdf_3hsp["ra"] = coord.ra.deg
    pdf_3hsp["dec"] = coord.dec.deg

    # format correctly names!
    pdf_3hsp["3HSP Source name"] = pdf_3hsp["3HSP Source name"].apply(
        lambda x: x.strip().replace('"', "")
    )

    return pdf_3hsp["ra"], pdf_3hsp["dec"], pdf_3hsp["3HSP Source name"]


def extract_gcvs(filename):
    """Read the gcvs catalog and extract useful columns

    Parameters
    ----------
    filename: str
        Path to the catalog (parquet file)

    Returns
    -------
    out: pd.Series, pd.Series, pd.Series
        ra, dec, VarType from the catalog

    Examples
    --------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> catalog = curdir + '/../../data/catalogs/gcvs.parquet'
    >>> ra, dec, vartype = extract_gcvs(catalog)
    """
    pdf = pd.read_parquet(filename)
    return pdf["ra"], pdf["dec"], pdf["VarType"]


def extract_vsx(filename):
    """Read the vsx catalog and extract useful columns

    Parameters
    ----------
    filename: str
        Path to the catalog (parquet file)

    Returns
    -------
    out: pd.Series, pd.Series, pd.Series
        ra, dec, VarType from the catalog

    Examples
    --------
    >>> import os
    >>> curdir = os.path.dirname(os.path.abspath(__file__))
    >>> catalog = curdir + '/../../data/catalogs/vsx.parquet'
    >>> ra, dec, vtype = extract_vsx(catalog)
    """
    pdf_vsx = pd.read_parquet(filename)
    return pdf_vsx["RAJ2000"], pdf_vsx["DEJ2000"], pdf_vsx["VType"]


@profile
def generate_csv(s: str, lists: list) -> str:
    """Make a string (CSV formatted) given lists of data and header.

    Parameters
    ----------
    s: str
        String which will contain the data.
        Should initially contain the CSV header.
    lists: list of lists
        List containing data.
        Length of `lists` must correspond to the header.

    Returns
    -------
    s: str
        Updated string with one row per line.

    Examples
    --------
    >>> header = "toto,tata\\n"
    >>> lists = [[1, 2], ["cat", "dog"]]
    >>> table = generate_csv(header, lists)
    >>> print(table)
    toto,tata
    1,"cat"
    2,"dog"
    <BLANKLINE>
    """
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
    _ = [writer.writerow(row) for row in zip(*lists)]
    return s + output.getvalue().replace("\r", "")


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    regular_unit_tests(globals())
