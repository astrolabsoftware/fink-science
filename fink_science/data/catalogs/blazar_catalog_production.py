# Copyright 2026 AstroLab Software
# Author: Julian Hamo
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

import io
import time
import logging
import argparse
import datetime
import requests
from pathlib import Path
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from astroquery.simbad import Simbad
#from ...ztf.blazar_extreme_state.utils import from_mag_to_flux

logger = logging.getLogger(__name__)

CATALOG_COLUMN_NAMES: list = [
    "Source_name",
    "ZTF_name",
    "medians",
    "low_threshold",
    "high_threshold",
]

BLAZAR_CLASSES: set = {"BLLac", "Blazar", "QSO"}
CANDIDATE_BLAZAR_CLASSES: set = {"Blazar_Candidate", "Radio", "NearIR"}
UNKNOWN_CLASSES: set = {"Unknown", "Unknown_Candidate"}

FINK_APIURL: str = "https://api.ztf.fink-portal.org"
DR_APIURL: str = "https://db.ztf.snad.space"
DR_CHECK_URL: str = "https://www.ztf.caltech.edu/nsf-msip.html"

search_radius: int = 2
dt_concomitance: float = 1 / 24
MAX_DT: float = 1
START_ZTF: float = 58000

CATALOG_FILEPATH: str = "./CTAO_blazars_ztf_dr23.v03_2026.parquet"
LOGDIRFILENAME: str = "blazar_watchlist.log"

def from_mag_to_flux(
    mag: np.ndarray, magerr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the flux, in Jansky, of the source from its DC magnitude.

    Parameters
    ----------
    mag : array_like
        DC magnitude of the source.
    magerr : array_like
        Uncertainties on the DC magnitude of the source.

    Returns
    -------
    out : pd.DataFrame
        Pandas DataFrame of the light curve with added computed flux.
    """
    flux = 3631 * 10 ** (-0.4 * mag)
    flux_error = flux * 0.4 * np.log(10) * magerr

    return flux, flux_error


# ===================================
# File handling
# ===================================


def read_file_names(path: str, colname: str = None) -> np.ndarray:
    """Read new source file.

    Read a file. Returns a list of names of sources. Handles multi-column
    tables if a key is given. Handles ``.parquet``, ``.csv``, ``.json`` and
    ``.pickle`` files.

    Parameters
    ----------
    path : string or Path
        Path of the catalog.
    colname : string, optional
        If given, uses this key to retrieve the names.
        Else uses the first column.

    Returns
    -------
    out : np.array
        Numpy array of the list of names for sources.
    """
    path = Path(path)
    if not path.is_file():
        logger.warning("File not found.")
        return np.array([])
    elif path.suffix == ".parquet":
        source_table = pd.read_parquet(path)
    elif path.suffix == ".csv":
        source_table = pd.read_csv(path)
    elif path.suffix == ".json":
        source_table = pd.read_json(path)
    elif path.suffix == ".pickle":
        source_table = pd.read_pickle(path)
    else:
        logger.warning("Extension not recognised.")
        return np.array([])
    if colname is not None and np.any(source_table.keys() == colname):
        return source_table[colname].to_numpy()
    return source_table[source_table.keys()[0]].to_numpy()


def _init_catalog() -> pd.DataFrame:
    """Subfunction to initialize an empty catalog.

    Returns
    -------
    out : pd.DataFrame
        Empty pandas DataFrame with the catalog schema.
    """
    return pd.DataFrame({key: [] for key in CATALOG_COLUMN_NAMES})


def read_catalog(path: str) -> pd.DataFrame:
    """Read existing catalog, if any.

    Read an already computed catalog. Returns it as a pandas DataFrame.
    If the catalog does not exists, initialize and empty DataFrame with
    ``CATALOG_COLUMN_NAMES``.

    Parameters
    ----------
    path : string or Path
        Path of the catalog.

    Returns
    -------
    out : pd.DataFrame
        Pandas DataFrame of the catalog
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".parquet":
        catalog = pd.read_parquet(path)
        if np.isin(catalog.keys(), CATALOG_COLUMN_NAMES).all():
            return catalog
    logger.warning("No formatted catalog found.")
    return _init_catalog()


def merge_catalog(listnames: np.ndarray, catalog: pd.DataFrame = None) -> pd.DataFrame:
    """Merge catalog and list of names of new sources.

    Add new sources to the catalog, returns a new DataFrame catalog.
    If no catalog, generates a fully new catalog. Produces the expected schema
    and fill the yet-to-be computed values with NaN.

    Parameters
    ----------
    listnames : array_like
        List of names of source to add to the catalog.
    catalog : pd.DataFrame, optional
        If given, DataFrame containing the formerly computed catalog.
        If not, initialized as empty.

    Returns
    -------
    out : pd.DataFrame
        New pandas DataFrame catalog containing the former catalog if given,
        and the new source rows.
    """
    if catalog is None:
        logger.debug("No catalog has been given.")
        catalog = _init_catalog()

    new_sources = pd.DataFrame({"Source_name": np.unique(listnames)})

    return pd.concat(
        [catalog[~catalog["Source_name"].isin(listnames)], new_sources],
        ignore_index=True,
    )


def write_catalog(catalog: pd.DataFrame, path: str) -> None:
    """Write catalog as a ``.parquet`` file.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog to write.
    path : string
        Path where to write the catalog.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        path = Path(f"{path}.parquet")
    catalog.to_parquet(path, index=False)
    return


def expand_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Expand catalog for one row per ZTF id.

    Create a catalog from the current catalog with a row per ZTF id
    instead of a list of ZTF ids per source.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog with ZTF ids.

    Returns
    -------
    out : pd.DataFrame
        Expanded catalog.
    """
    list_to_concat = []
    for _, row in catalog.iterrows():
        ztf_ids = row["ZTF_name"]
        if isinstance(ztf_ids, (list, np.ndarray)) and len(ztf_ids):
            for ztf_id in ztf_ids:
                new_row = row.copy()
                new_row["ZTF_name"] = ztf_id
                list_to_concat.append(new_row.to_frame().T)
        else:
            new_row = row.copy()
            new_row["ZTF_name"] = ""
            list_to_concat.append(new_row.to_frame().T)

    return pd.concat(list_to_concat, ignore_index=True)


# =========================
# DR and Fink API handling
# =========================


def check_dr_version() -> int:
    """Scrape the ZTF DR website to get the latest public DR version.

    Returns
    -------
    dr_version : int
        Latest Data Release number. Returns -1 if it cannot be determined.
    """
    try:
        r = requests.get(DR_CHECK_URL)
        lines = r.text.split("\n")
        detection_key = '\t\t\t\t\t<table class="table table-bordered">'
        table_index = np.where(np.array(lines) == detection_key)[0][0]
        dr_line = lines[table_index + 9].split("<")[4]
        dr_version = int(dr_line[23:])
        return dr_version
    except Exception:
        logger.warning("check_DR_version failed: {e}")
        return -1


def _post_fink(
    url: str,
    payload: dict,
    max_retries: int = 10,
    delay: float = 0.5,
) -> requests.Response:
    """Perform a POST request with retry logic.

    Parameters
    ----------
    url : str
        Endpoint URL.
    payload : dict
        Request payload.
    max_retries : int, optional
        Maximum number of retries. Default is 10.
    delay : float, optional
        Delay (in seconds) between retries. Default is 0.5.

    Returns
    -------
    response : requests.Response
        Successful response object.

    Raises
    ------
    ConnectionError
        If the request fails after all retries.
    """
    for attempt in range(max_retries):
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response
        else:
            time.sleep(delay)
            delay *= 2
            logger.warning(f"_post_fink failed (attempt {attempt + 1})")

    # Error logger handling
    if response.status_code != 200:
        logger.exception(f"Failed to connect to {url} after {max_retries} retries.")
        raise


def _get_snad(
    url: str,
    payload: dict,
    max_retries: int = 10,
    delay: float = 0.5,
) -> requests.Response:
    """Perform a POST request with retry logic.

    Parameters
    ----------
    url : str
        Endpoint URL.
    payload : dict
        Request payload.
    max_retries : int, optional
        Maximum number of retries. Default is 10.
    delay : float, optional
        Delay (in seconds) between retries. Default is 0.5.

    Returns
    -------
    response : requests.Response
        Successful response object.

    Raises
    ------
    ConnectionError
        If the request fails after all retries.
    """
    for attempt in range(max_retries):
        response = requests.get(url, params=payload, timeout=10)
        if response.status_code == 200:
            return response
        else:
            time.sleep(delay)
            delay *= 2
            logger.warning(f"_post_fink failed (attempt {attempt + 1})")

    # Error logger handling
    if response.status_code != 200:
        logger.exception(f"Failed to connect to {url} after {max_retries} retries.")
        raise


def get_simbad_coordinates(catalog: pd.DataFrame) -> pd.DataFrame:
    """Find SIMBAD coordinates.

    Query CDS SIMBAD API to retrieve their equatorial coordinates for a
    the sources of the catalog. Returns (np.nan, np.nan) if no coordinates has
    been found for a given source. Returns the updated catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        DataFrame containing the names of the sources.

    Returns
    -------
    catalog : pd.DataFrame
        Pandas DataFrame of the updated catalog with right ascension and
        declination.

    Raises
    ------
    NameError
        If no sources has been found in the CDS SIMBAD database.
    """
    simbad_df = Simbad.query_objects(catalog["Source_name"].to_numpy()).to_pandas()

    if simbad_df.empty:
        logger.error("No sources found in CDS SIMBAD for the catalog.")
        raise

    catalog = catalog.merge(
        simbad_df[["user_specified_id", "ra", "dec"]],
        left_on="Source_name",
        right_on="user_specified_id",
        how="left",
    )
    return catalog


def _get_fink_data(name: str) -> pd.DataFrame:
    """Get alert packet from Fink.

    Retrieve the full alert history for a given ZTF object from the Fink
    database.

    Parameters
    ----------
    name : str
        ZTF object identifier (e.g., "ZTF19aapreis").

    Returns
    -------
    lc : pandas.DataFrame
        DataFrame containing the alert history of the object, with:
        - `i:fid` : Filter identifier (1=g, 2=r, 3=i)
        - Other alert metadata provided by Fink

    Raises
    ------
    NameError
        If the source is not found in the Fink database.

    Notes
    -----
    - Only alerts tagged as "valid" are returned.
    - The returned DataFrame is sorted in descending order of `i:mjd`.
    - The request is retried up to 10 times if the Fink API does not respond.

    Examples
    --------
    >>> df = get_Fink_data("ZTF19aapreis")
    >>> df[["i:mjd", "i:fid", "i:magpsf"]].head()
    """
    response = _post_fink(
        f"{FINK_APIURL}/api/v1/objects",
        payload={
            "objectId": name,
            "output-format": "json",
            "withupperlim": "True",
        },
    )

    lc = pd.read_json(io.BytesIO(response.content))
    if lc.empty:
        logger.warning(f'Source "{name}" not found in Fink.')

    lc["i:mjd"] = lc["i:jd"] - 2400000.5
    lc["i:fid"] = lc["i:fid"].astype(int)
    lc = lc[lc["d:tag"] == "valid"].sort_values(
        "i:mjd", ascending=True, ignore_index=True
    )
    return lc


def _get_class_ztf_identifier(
    ra: float,
    dec: float,
    radius: float,
    source_classes: set = BLAZAR_CLASSES,
    candidate_source_classes: set = CANDIDATE_BLAZAR_CLASSES,
    unknown_classes: set = UNKNOWN_CLASSES,
) -> np.ndarray:
    """Retrieve ZTF identifier for a given position.

    Retrieve ZTF identifiers near a position, prioritizing blazar-like
    classifications.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    radius : float, optional
        Cone search radius in arcseconds.
    source_classes : set of str, optional
        Fink classes identifying confirmed blazars.
    candidate_source_classes : set of str, optional
        Classes for possible misclassifications (blazar candidates).
    unknown_classes : set of str, optional
        Classes used when the object is unclassified.

    Returns
    -------
    np.ndarray
        Array of matching ZTF identifiers. Empty array if no match found.
    """
    r = _post_fink(
        f"{FINK_APIURL}/api/v1/conesearch",
        payload={"ra": ra, "dec": dec, "radius": radius, "columns": "i:objectId"},
    )

    lc = pd.read_json(io.BytesIO(r.content))
    if lc.empty:
        logger.debug(f"No found Fink correspondance for ra={ra:.6f}, dec={dec:.6f}.")
        return np.array([])

    names = lc["i:objectId"].unique()
    lcs = [_get_fink_data(name) for name in names]

    # Convert classifications into sets for faster intersection
    classifications = [set(lc["v:classification"].unique()) for lc in lcs]

    # Step 1: confirmed blazars
    tags = np.array([bool(classes & source_classes) for classes in classifications])
    if tags.any():
        logger.debug("Source confirmed as blazar in Fink classification.")
        return names[tags]

    # Step 2: candidate classes
    tags = np.array([
        bool(classes & candidate_source_classes) for classes in classifications
    ])
    if tags.any():
        logger.debug("Source candidate blazar in Fink classification.")
        return names[tags]

    # Step 3: unknown only if all classifications are unknown
    tags = np.array([classes.issubset(unknown_classes) for classes in classifications])
    logger.debug("Source classification not known by Fink.")
    return names[tags]


def get_ztf_id(catalog: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Retrieve ZTF ids for all catalog.

    Retrieve list of ZTF ids used in Fink that correspond to a source from
    a cone search around its coordinates. Returns an empty list if nothing has
    been found. Returns the updated catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        DataFrame containing the coordinates of the sources.
    radius : float
        radius (in arcseconds) to use in cone search.

    Returns
    -------
    catalog : pd.DataFrame
        Pandas DataFrame of the updated catalog with ZTF identifiers.
    """
    ztf_ids = []

    if not np.isin(["ra", "dec"], catalog.keys()).all():
        raise KeyError("get_ztf_id failed - no 'ra' or 'ded' found in keys.")

    for index, row in catalog.iterrows():
        name = row["Source_name"]
        logger.debug(
            f"Retrieving ZTF id for source {name} ({index + 1}/{len(catalog)})"
        )
        ztf_ids.append(
            _get_class_ztf_identifier(ra=row["ra"], dec=row["dec"], radius=radius)
        )
    catalog["ZTF_name"] = ztf_ids
    return catalog


# DR download within 2"-cone search


def _get_ztf_dr_data(ra: float, dec: float, radius: float) -> pd.DataFrame:
    """Retrieve ZTF light curves from the latest Data Release via SNAD API.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    radius : float
        Search radius in arcseconds.

    Returns
    -------
    pd.DataFrame
        Light curve data including fluxes,
        uncertainties, filters, and metadata.
    """
    response = _get_snad(
        f"{DR_APIURL}/api/v3/data/latest/circle/full/json",
        payload={"ra": ra, "dec": dec, "radius_arcsec": radius},
    )

    records = []
    for oid, entry in response.json().items():
        lc = pd.DataFrame(entry["lc"])

        # Common metadata
        meta = entry["meta"]
        base_meta = {
            "filtercode": meta["filter"],
            "ra": meta["coord"]["ra"],
            "dec": meta["coord"]["dec"],
            "fieldid": meta["fieldid"],
            "ngoodobs": meta["ngoodobs"],
            "duration": meta["duration"],
            "rcid": meta["rcid"],
            "nobs": meta["nobs"],
            "oid": oid,
        }

        # Add h3 meta values with prefix
        h3_meta = {f"h3_{k}": v for k, v in meta["h3"].items()}

        # Repeat metadata for each row of lc1
        for key, val in {**base_meta, **h3_meta}.items():
            lc[key] = val

        records.append(lc)

    if not records:
        logger.debug("No light curve found in DR.")
        return pd.DataFrame()

    lc = pd.concat(records, ignore_index=True)

    filter_map = {"zg": 1, "zr": 2, "zi": 3}
    lc["filtercode"] = lc["filtercode"].map(filter_map).astype(int)
    lc = lc[(lc["mjd"] >= START_ZTF) & np.isin(lc["filtercode"], [1, 2])].copy()
    lc = lc.sort_values("mjd", ascending=True, ignore_index=True)

    return lc


def get_ztf_dr_data(catalog: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Retrieve DR light curves.

    Download the light curve of the sources of the catalog from a
    conesearch around their coordinates. Returns an empty DataFrame if no
    measurement has been found in the cone search. Returns the updated
    catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        DataFrame containing the coordinates of the sources.
    radius : float
        radius (in arcseconds) to use in cone search.

    Returns
    -------
    catalog : pd.DataFrame
        Pandas DataFrame of the updated catalog with the ZTF DR light curves.
    """
    lcs = []
    for index, row in catalog.iterrows():
        name = row["Source_name"]
        logger.debug(
            f"Retrieving ZTF DR light curve \
for source {name} ({index + 1}/{len(catalog)})"
        )
        lcs.append(_get_ztf_dr_data(ra=row["ra"], dec=row["dec"], radius=radius))
    catalog["ZTF_lc"] = lcs
    return catalog


# ===================================
# Standardisation of the light curve
# ===================================


# 1 band standardisation


def _standardise_lc_1band(lc: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Standardise 1-band light curve flux.

    Parameters
    ----------
    lc : pd.DataFrame
        Pandas DataFrame of the light curve to be standardised.
        Must contain: ``mjd``, ``flux``, ``flux_error``.

    Returns
    -------
    out : Tuple of (pd.DataFrame, float)
        Pandas DataFrame of the light curve with added standardised flux
        and median used for standardisation.

    Notes
    -----
    Standardisation process refers to division of the flux by the median of a
    meaningful subset of measurements. Here we compute the median of the light
    curve weighted by the inverse of its uncertainties and the elapsed time
    until the next measurement.
    """
    time = lc["mjd"].to_numpy()
    flux = lc["flux"].to_numpy()
    flux_error = lc["flux_error"].to_numpy()

    if len(flux) < 2:
        median = flux[0]
    else:
        diff_time = np.diff(time)
        diff_time[diff_time > MAX_DT] = MAX_DT
        median = float(
            np.quantile(
                flux[:-1],
                0.5,
                method="inverted_cdf",
                weights=diff_time / (flux_error[:-1] ** 2),
            )
        )

    lc.loc[:,"std_flux"] = flux / median
    return lc, median


# 2 bands standardisation


def _concomitant_weighted(
    t1: np.ndarray,
    f1: np.ndarray,
    e1: np.ndarray,
    t2: np.ndarray,
    f2: np.ndarray,
    e2: np.ndarray,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the concomitant matching of two light curves.

    Both band 1 and band 2 fluxes can have multiple points in a time window T.
    For each concomitant group, weighted averages are computed.
    Band 1 and band 2 are arbitrary labels, they don't need to be associated
    to a specific filter.

    Parameters
    ----------
    t1, f1, e1 : array_like
        Times, fluxes, and errors for flux of band 1.
    t2, f2, e2 : array_like
        Times, fluxes, and errors for flux of band 2.
    T : float
        Concomitance time window (same units as t1 and t2).

    Returns
    -------
    f1_out, f2_out : np.ndarray
        Matched weighted fluxes for band 1 and band 2.
    """
    f1_out, f2_out = [], []

    for t in t1:
        mask2 = np.abs(t2 - t) <= T
        if mask2.sum():
            f2_mean = np.average(f2[mask2], weights=1 / e2[mask2] ** 2)
            mask1 = np.abs(t1 - t) <= T
            f1_mean = np.average(f1[mask1], weights=1 / e1[mask1] ** 2)
            f1_out.append(f1_mean)
            f2_out.append(f2_mean)

    return np.array(f1_out), np.array(f2_out)


def _standardise_lc_2bands(
    lc: pd.DataFrame, dt_concomitance: float
) -> tuple[pd.DataFrame, dict]:
    """Standardise 2-band light curve flux.

    Parameters
    ----------
    lc : pd.DataFrame
        Pandas DataFrame of the light curve to be standardised.
        Must contain: ``mjd``, ``flux``, ``flux_error``.
    dt_concomitance : float
        Delta time used to select concomitant measurements
        in the case of 2-band light curves.

    Returns
    -------
    out : Tuple of (pd.DataFrame, dict)
        Pandas DataFrame of the light curve with added standardised flux and
        medians used for standardisation.

    Notes
    -----
    Standardisation process refers to division of the flux by the median of a
    meaningful subset of measurements. Here we select interband measurements
    that are at most ``dt_concomitance`` apart. If necessary, we average
    measurements within a band to compare only one measurement per concomitant
    detection.
    """
    time = lc["mjd"].to_numpy()
    measurements = lc["flux"].to_numpy()
    uncertainties = lc["flux_error"].to_numpy()
    filters = lc["filtercode"].to_numpy()
    mask = filters == filters[0]
    unique_filters = np.unique(filters).astype(str)
    medians = {filt: np.nan for filt in unique_filters}
    lc["std_flux"] = np.full(len(lc), np.nan)

    # Compute concomitant measurements
    (
        lc1_concomitant,
        lc2_concomitant,
    ) = _concomitant_weighted(
        time[mask],
        measurements[mask],
        uncertainties[mask],
        time[~mask],
        measurements[~mask],
        uncertainties[~mask],
        dt_concomitance,
    )

    # Case 1: there are concomitant measurements
    if len(lc1_concomitant):
        lc.loc[mask, "std_flux"] = lc.loc[mask, "flux"] / np.nanmedian(lc1_concomitant)
        lc.loc[~mask, "std_flux"] = lc.loc[~mask, "flux"] / np.nanmedian(
            lc2_concomitant
        )
        medians = {
            unique_filters[0]: np.nanmedian(lc1_concomitant),
            unique_filters[1]: np.nanmedian(lc2_concomitant),
        }

    # Case 2 : No concomitant measurements were found
    else:
        tmp, median1 = _standardise_lc_1band(lc[mask])
        lc.loc[mask, "std_flux"] = tmp["std_flux"].to_numpy()
        tmp, median2 = _standardise_lc_1band(lc[~mask])
        lc.loc[~mask, "std_flux"] = tmp["std_flux"].to_numpy()
        medians = {unique_filters[0]: median1, unique_filters[1]: median2}

    return lc, medians


# Full band handling standardisation


def medians_for_1band(lc: pd.DataFrame, median: float) -> dict:
    """Subfunction helper to format median dictionary for catalog.

    Parameters
    ----------
     lc : pd.DataFrame
        Pandas DataFrame of the light curve to be standardised.
        Must contain: ``mjd``, ``flux``, ``flux_error``, ``filtercode``.
    median : float
        Median computed to use for standardisation.

    Returns
    -------
    medians : dict
        Dictionary of per-band medians. Filled with nan for not known values.
    """
    medians = {"1": np.nan, "2": np.nan}
    medians[str(lc["filtercode"].iloc[0])] = median
    return medians


def standardise_lc(
    lc: pd.DataFrame, dt_concomitance: float
) -> tuple[pd.DataFrame, dict]:
    """Standardise a light curve flux. Handle 1- and 2-bands cases.

    Parameters
    ----------
    lc : pd.DataFrame
        Pandas DataFrame of the light curve to be standardised.
        Must contain: ``mjd``, ``flux``, ``flux_error``, ``filtercode``.
    dt_concomitance : float
        Delta time used to select concomitant measurements
        in the case of 2-band light curves.

    Returns
    -------
    out : Tuple of (pd.DataFrame, dict)
        Pandas DataFrame of the light curve with added standardised flux
        and tuple of medians for further standardisation.

    Notes
    -----
    This method uses subfunctions ``standardise_lc_1band`` and
    ``standardise_lc_2bands``. Please refer to their documentation for
    complementary details.
    """
    if not np.isin(["flux", "flux_error"], lc.keys()).all():
        lc["flux"], lc["flux_error"] = from_mag_to_flux(
            lc["mag"].to_numpy(), lc["magerr"].to_numpy()
        )
    if len(lc["filtercode"].unique()) == 1:
        lc, median = _standardise_lc_1band(lc)
        medians = medians_for_1band(lc, median)
        return lc, medians
    else:
        return _standardise_lc_2bands(lc, dt_concomitance)


def standardise_lcs(catalog: pd.DataFrame, dt_concomitance: float) -> pd.DataFrame:
    """Standardise all light curves from catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        DataFrame containing the light curves of the sources.
    dt_concomitance : float
        Delta time used to select concomitant measurements
        in the case of 2-band light curves.

    Returns
    -------
    catalog : pd.DataFrame
        Pandas DataFrame of the updated catalog with the ZTF DR light curves
        and medians for further standardisation.
    """
    lcs = []
    medians = []

    if "ZTF_lc" not in catalog.keys():
        raise KeyError("standardise_lcs failed: no 'ZTF_lc' found in catalog keys.")

    for index, row in catalog.iterrows():
        name = row["Source_name"]
        logger.debug(
            f"Standardising light curve of \
source {name} ({index + 1}/{len(catalog)})"
        )
        lc = row["ZTF_lc"].copy()
        if lc.empty:
            logger.warning("Empty light curve.")
            lc["std_flux"] = np.nan
            medians_source = {"1": np.nan, "2": np.nan}
        else:
            lc, medians_source = standardise_lc(lc, dt_concomitance)
        lcs.append(lc)
        medians.append(medians_source)
    catalog["ZTF_lc"] = lcs
    catalog["medians"] = medians
    return catalog


# =====================
# Quantity computation
# =====================

# Flux quantile computation


def compute_threshold(
    lc: pd.DataFrame, high_threshold: float, low_threshold: float
) -> tuple[np.float64, np.float64]:
    """Compute the flux thresholds for one light curve.

    Compute the flux thresholds on the cumulated distribution function
    (CDF) of the light curve.

    Parameters
    ----------
    lc : pd.DataFrame
        Pandas DataFrame of the Data Release light curve containing at least:
        ``mjd``, ``std_flux``.
    high_threshold : float
        Flux quantile to determine high state threshold.
    low_threshold : float
        Flux quantile to determine low state threshold.

    Returns
    -------
    out : tuple of np.float64
        High and low thresholds (in that order) from CDF of the standardised
        flux.

    Notes
    -----
    Every flux measurement used for the construction of the CDF is weighted by
    dt, where dt is the elapsed time between the considered measurement and
    the next one. We clip dt to 1 to avoid abnormal weights due to
    observational gaps (we keep it to 1 day at most).
    """
    time = lc["mjd"].to_numpy()
    measurements = lc["std_flux"].to_numpy()

    weights = np.diff(time)
    weights[weights > MAX_DT] = MAX_DT
    measurements = measurements[:-1]

    idx = np.argsort(measurements)
    sort_measurements = measurements[idx]
    sort_weights = weights[idx]

    if np.sum(sort_weights) == 0:
        logger.warning("Empty light curve.")
        return np.nan, np.nan
    cdf = np.cumsum(sort_weights) / np.sum(sort_weights)

    return (
        np.interp(
            high_threshold,
            cdf,
            sort_measurements,
            left=sort_measurements[0],
            right=sort_measurements[-1],
        ),
        np.interp(
            low_threshold,
            cdf,
            sort_measurements,
            left=sort_measurements[0],
            right=sort_measurements[-1],
        ),
    )


def compute_quantile_for_catalog(
    catalog: pd.DataFrame, high_threshold: float = 0.9, low_threshold: float = 0.1
) -> pd.DataFrame:
    """Compute flux thresholds of all light curves.

    Compute the flux thresholds on the cumulated distribution function
    (CDF) of the light curve. Returns updated catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        DataFrame containing the light curves of the sources.
    high_threshold : float, optional
        Flux quantile to determine high state threshold.
        Default is 0.9.
    low_threshold : float, optional
        Flux quantile to determine low state threshold.
        Default is 0.1.

    Returns
    -------
    catalog : pd.DataFrame
        Pandas DataFrame of the updated catalog with the thresholds for
        extreme states.
    """
    low_thresholds = []
    high_thresholds = []

    if "ZTF_lc" not in catalog.keys():
        raise KeyError(
            "compute_quantile_for_catalog failed: \
no 'ZTF_lc' found in catalog keys."
        )

    for index, row in catalog.iterrows():
        name = row["Source_name"]
        logger.debug(
            f"Computing thresholds \
for source {name} ({index + 1}/{len(catalog)})"
        )
        lc = catalog["ZTF_lc"].iloc[index].copy()
        if lc.empty:
            logger.warning("Empty light curve.")
            low_thresholds.append(np.nan)
            high_thresholds.append(np.nan)
        else:
            high_thres, low_thres = compute_threshold(lc, high_threshold, low_threshold)
            low_thresholds.append(low_thres)
            high_thresholds.append(high_thres)

    catalog["low_threshold"] = low_thresholds
    catalog["high_threshold"] = high_thresholds
    return catalog


# ===========
# Main + CLI
# ===========


def setup_logging(
    log_filepath: str = LOGDIRFILENAME, level: int = logging.INFO
) -> logging.Logger:
    """Configure logging for console and file output.

    Parameters
    ----------
    log_file_name : str
        Path of the log file.
    level : int
        Logging level (logging.INFO, logging.DEBUG, etc.)

    Returns
    -------
    logger : logging.Logger
        Logger to reference activity of the pipeline.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Prevent duplicate handlers if re-run
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # Rotating file handler
    file_handler = RotatingFileHandler(log_filepath, maxBytes=5_000_000, backupCount=3)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    out : argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fink blazar watchlist production pipeline"
    )

    parser.add_argument(
        "--source-list",
        nargs="?",
        const="",
        default="",
        help="Path for the list of sources from which create the catalog",
    )

    parser.add_argument(
        "--source-key",
        nargs="?",
        const=CATALOG_COLUMN_NAMES[0],
        default=CATALOG_COLUMN_NAMES[0],
        help="Key for the list of sources",
    )

    parser.add_argument(
        "--catalog-path",
        nargs="?",
        const=None,
        default=None,
        help="Path where to save the catalog",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--search-radius",
        nargs="?",
        const=search_radius,
        default=search_radius,
        help="Search radius (in arcseconds) for Fink and SNAD API",
    )

    parser.add_argument(
        "--dt-concomitance",
        nargs="?",
        const=dt_concomitance,
        default=dt_concomitance,
        help="Delta time (in days) to assume concomitance of measurements",
    )

    return parser.parse_args()


def log_execution_time(start_time):
    """Compute execution time from the start.

    Parameters
    ----------
    start_time : float
        Time when the pipeline started.

    Returns
    -------
    h, m, s : ints
        Elapsed time, returned in hour, minutes and seconds, respectively.
    """
    elapsed = int(time.time() - start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return h, m, s


def main():
    """main() blazar watchlist production pipeline process."""
    start_time = time.time()
    args = parse_args()

    # Start logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)

    logger.info("Start of the blazar watchlist production process.")

    # Configurate file name of the catalog
    catalog_filepath = args.catalog_path
    if catalog_filepath is None:
        version = check_dr_version()
        now = datetime.datetime.now()
        date = f"{str(now.month).zfill(2)}_{now.year}"
        catalog_filepath = f"./CTAO_blazars_ztf_dr{version}.v{date}"
        catalog_filepath += ".parquet"

    # Retrieve list of sources
    list_names = read_file_names(args.source_list, colname=args.source_key)
    logger.info(f"Number of sources waiting to be added: {len(list_names)}")

    # Load catalog, if exists
    logger.info("Browsing to find former catalog")
    catalog = read_catalog(catalog_filepath)

    # Merge list and catalog
    catalog = merge_catalog(list_names, catalog)
    logger.info(f"Full shape of the new catalog: {catalog.shape}")

    # Retrieve Simbad coordinates
    logger.info("Retrieving source coordinates in SIMBAD database")
    catalog = get_simbad_coordinates(catalog)

    # Search ZTF ids
    logger.info("Retrieving ZTF identifiers of the catalog sources")
    catalog = get_ztf_id(catalog, radius=search_radius)

    # Download ZTF DR light curves
    logger.info("Retrieving DR light curves of the catalog sources")
    catalog = get_ztf_dr_data(catalog, radius=search_radius)

    h, m, s = log_execution_time(start_time)
    logger.info(f"Total elapsed time after download: {h}h {m}m {s}s")

    # Standardise ZTF light curves
    logger.info("Standardising light curves of the catalog sources")
    catalog = standardise_lcs(catalog, dt_concomitance)

    # Compute low- and high-threshold values
    logger.info("Computing thresholds for the catalog sources")
    catalog = compute_quantile_for_catalog(
        catalog, high_threshold=0.9, low_threshold=0.1
    )
    try:
        logger.info("Catalog satisfies expected scheme")
        assert set(CATALOG_COLUMN_NAMES).issubset(catalog.columns)
    except Exception:
        logger.warning("main failed - wrong scheme for catalog: {e}")

    h, m, s = log_execution_time(start_time)
    logger.info(f"Total elapsed time after analysis: {h}h {m}m {s}s")

    # Expand catalog
    catalog = expand_catalog(catalog[CATALOG_COLUMN_NAMES])
    logger.info(f"Final shape of the expanded catalog: {catalog.shape}")

    # Write catalog
    write_catalog(catalog, catalog_filepath)
    logger.info(f"Catalog written at {catalog_filepath}")

    h, m, s = log_execution_time(start_time)
    logger.info(f"Total execution time: {h}h {m}m {s}s")


# Execute main

if __name__ == "__main__":
    main()
