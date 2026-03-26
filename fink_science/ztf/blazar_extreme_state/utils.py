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

import requests
import time
import logging

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


# ============================
# Estimation of extreme state
# ============================


def _instantness_criterion(
    pdf: pd.DataFrame, CTAO_blazar: pd.DataFrame, state_key: str
) -> np.float64:
    """Compute instantness criterion for a given state.

    Returns the standardised flux of the last measurement
    over the precomputed threshold ratio.

    Parameters
    ----------
    pdf : pd.DataFrame
        Pandas DataFrame of the alert history containing at least:
        ``candid``, ``ojbectId``, ``cdistnr``, ``cmagpsf``, ``csigmapsf``,
        ``cmagnr``, ``csigmagnr``, ``cisdiffpos``, ``cfid``, ``cjd``,
        ``cstd_flux``, ``csigma_std_flux``.
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.
    state_key : string
        Key for the threshold to retrieve from ``CTAO_blazar``.
        Either ``low_threshold`` or ``high_threshold``.

    Returns
    -------
    out : np.float64
        Ratio of the standardised flux coming from the last measurement alert
        over precomputed threshold.
    """
    name = pdf["objectId"].to_numpy()[0]

    try:
        threshold = np.array(
            CTAO_blazar.loc[CTAO_blazar["ZTF_name"] == name, state_key].to_numpy()[0]
        )
    except IndexError as e:
        _LOG.warning(f"_instantness_criterion process failed: {e}.")
        return -1

    try:
        return pdf["cstd_flux"].iloc[-1] / threshold
    except KeyError as e:
        _LOG.warning(f"_instantness_criterion process failed: {e}.")
        return -1


def _robustness_criterion(
    pdf: pd.DataFrame,
    CTAO_blazar: pd.DataFrame,
    state_key: str,
    integration_period: float,
) -> np.float64:
    """Compute robustness criterion for a given state.

    Returns the sliding mean over 30 days of the standardised flux
    over the precomputed threshold ratio.

    Parameters
    ----------
    pdf : pd.DataFrame
        Pandas DataFrame of the alert history containing at least:
        ``candid``, ``ojbectId``, ``cdistnr``, ``cmagpsf``, ``csigmapsf``,
        ``cmagnr``, ``csigmagnr``, ``cisdiffpos``, ``cfid``, ``cjd``,
        ``cstd_flux``, ``csigma_std_flux``.
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.
    state_key : string
        Key for the threshold to retrieve from ``CTAO_blazar``.
        Either ``low_threshold`` or ``high_threshold``.

    Returns
    -------
    out: np.float64
        Ratio of the sliding mean over 30 days of the standardised flux over
        the precomputed threshold
    """
    integration_period = 30
    name = pdf["objectId"].to_numpy()[0]

    try:
        threshold = np.array(
            CTAO_blazar.loc[CTAO_blazar["ZTF_name"] == name, state_key].to_numpy()[0]
        )
    except IndexError as e:
        _LOG.warning(f"_robustness_criterion process failed: {e}.")
        return -1

    try:
        full_time = pdf["cjd"]
        maskTime = full_time >= full_time.iloc[-1] - integration_period
        time = pdf.loc[maskTime, "cjd"]
        flux = pdf.loc[maskTime, "cstd_flux"]
    except KeyError as e:
        _LOG.warning(f"_robustness_criterion process failed: {e}.")
        return -1

    maskNan = pd.notna(flux)
    mtime = time[maskNan]
    if maskNan.sum() > 1:
        mtimestart = mtime.iloc[0]
        mtimestop = mtime.iloc[-1]
        integral = np.trapezoid(flux[maskNan], x=mtime)
        return integral / (mtimestop - mtimestart) / threshold
    else:
        _LOG.warning(
            "_robustness_criterion process failed:\
not enough points to compute fluence."
        )
        return -1


def extreme_state_(
    pdf: pd.DataFrame,
    CTAO_blazar: pd.DataFrame,
    state_key: str,
    integration_period: float,
) -> np.ndarray:
    """Returns an array containing blazar features.

    Parameters
    ----------
    pdf : pd.DataFrame
        Pandas DataFrame of the alert history containing at least:
        ``candid``, ``ojbectId``, ``cdistnr``, ``cmagpsf``, ``csigmapsf``,
        ``cmagnr``, ``csigmagnr``, ``cisdiffpos``, ``cfid``, ``cjd``,
        ``cstd_flux``, ``csigma_std_flux``.
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.
    state_key : string
        Key for the threshold to retrieve from ``CTAO_blazar``.
        Either ``low_threshold`` or ``high_threshold``.
    integration_period : float
        Integration period for the computation of the fluence in the
        robustness criterion.

    Returns
    -------
    out: np.ndarray of np.float64
        Array of ratios for:\n
        \t- Mean over threshold of the last alert\n
        \t- Measurement over threshold of the last alert

    Notes
    -----
    Features are:
        instantness: The mean over threshold ratio of the last alert
        robustness: The standardised flux over threshold ratio of the last
        alert
    """
    name = pdf["objectId"].to_numpy()[0]
    _LOG.info(f"Extreme state determination of {name}.")

    if not CTAO_blazar.loc[CTAO_blazar["ZTF_name"] == name].empty:
        return np.array([
            _robustness_criterion(pdf, CTAO_blazar, state_key, integration_period),
            _instantness_criterion(pdf, CTAO_blazar, state_key),
        ])

    else:
        return np.full(2, -1)


# ============================
# Download of the light curve
# ============================


def _post_request_with_retry(
    url: str,
    payload: dict,
    max_retries: int = 3,
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
        Maximum number of retries. Default is 3.
    delay : float, optional
        Delay (in seconds) between retries. Default is 0.5.

    Returns
    -------
    response : requests.Response
        Successful response object. Else None.
    """
    for _ in range(max_retries):
        response = requests.get(url, params=payload, timeout=10)
        if response.status_code == 200:
            return response
        else:
            time.sleep(delay)

    _LOG.warning(f"Failed to connect to {url} after {max_retries} retries.")
    return None


def get_ztf_dr_data(ra: float, dec: float, radius: float) -> pd.DataFrame:
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
    _LOG.info(f"DR light curve download of ra={ra:6f}, dec={dec:.6f}.")
    DR_APIURL = "https://db.ztf.snad.space"
    response = _post_request_with_retry(
        f"{DR_APIURL}/api/v3/data/latest/circle/full/json",
        payload={"ra": ra, "dec": dec, "radius_arcsec": radius},
    )
    if response is None:
        _LOG.info("Failed to retrieve the DR light curve.")
        return pd.DataFrame()

    records = []
    for oid, entry in response.json().items():
        lc1 = pd.DataFrame(entry["lc"])

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
            lc1[key] = val

        records.append(lc1)

    if not records:
        _LOG.warning("Retrieved empty light curve.")
        return pd.DataFrame()

    pdf = pd.concat(records, ignore_index=True)

    filter_map = {"zg": 1, "zr": 2, "zi": 3}
    pdf["filtercode"] = pdf["filtercode"].map(filter_map).astype(int)
    pdf = pdf[(pdf['mjd'] >= 58000) & np.isin(pdf["filtercode"], [1, 2])].copy()
    pdf = pdf.sort_values('mjd', ascending=True, ignore_index=True)

    return pdf


# ===================================
# Standardisation of the light curve
# ===================================


def from_mag_to_flux(lc: pd.DataFrame) -> pd.DataFrame:
    """Compute the flux, in Jansky, of the source from its DC magnitude.

    Parameters
    ----------
    lc : pd.DataFrame
        Pandas DataFrame of the light curve to be converted.
        Must contain: ``mjd``, ``mag``, ``magerr``.

    Returns
    -------
    out : pd.DataFrame
        Pandas DataFrame of the light curve with added computed flux.
    """
    measurements = lc["mag"].to_numpy()
    uncertainties = lc["magerr"].to_numpy()

    lc["flux"] = 3631 * 10 ** (-0.4 * measurements)
    lc["flux_error"] = lc["flux"].to_numpy() * 0.4 * np.log(10) * uncertainties

    return lc


def standardise_lc(
    pdf: pd.DataFrame, lc: pd.DataFrame, CTAO_blazar: pd.DataFrame
) -> pd.DataFrame:
    """Standardise a light curve using previously computed per-band medians.

    Parameters
    ----------
    pdf : pd.DataFrame
        Pandas DataFrame of the alert history containing at least:
        ``candid``, ``ojbectId``, ``cdistnr``, ``cmagpsf``, ``csigmapsf``,
        ``cmagnr``, ``csigmagnr``, ``cisdiffpos``, ``cfid``, ``cjd``,
        ``cstd_flux``, ``csigma_std_flux``.
    lc : pd.DataFrame
        Pandas DataFrame of the light curve to be converted.
        Must contain: ``filtercode``, ``flux``.
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.

    Returns
    -------
    out : pd.DataFrame
        Pandas DataFrame of the light curve with added standardised flux.
    """
    name = pdf["objectId"].to_numpy()[0]
    _LOG.info(f"Standardisation of {name}")

    sub_catalog = CTAO_blazar[CTAO_blazar["ZTF_name"] == name]
    lc["std_flux"] = np.full(len(lc), np.nan)
    for filt in lc["filtercode"].unique():
        maskFilt = lc["filtercode"] == filt
        median = sub_catalog["medians"][filt]
        lc.loc[maskFilt, "std_flux"] = lc.loc[maskFilt, "flux"] / median
    return lc


# ==================================
# Determination of the CDF quantile
# ==================================


def compute_quantile(lc: pd.DataFrame, measurement: np.float64) -> np.float64:
    """Place the measurement on the CDF.

    Compute the quantile on the cumulated distribution function (CDF)
    of the light curve corresponding to the given measurement.

    Parameters
    ----------
    lc : pd.DataFrame
        Pandas DataFrame of the Data Release light curve containing at least:
        ``mjd``, ``std_flux``.
    measurement : np.float64
        Measurement in standardised flux units.

    Returns
    -------
    quantile : np.float64
        Quantile corresponding to ``measurement``
        on the CDF of the flux of the source.
    """
    time = lc["mjd"].to_numpy()
    measurements = lc["std_flux"].to_numpy()

    weights = np.diff(time)
    measurements = measurements[:-1]

    idx = np.argsort(measurements)
    sort_measurements = measurements[idx]
    sort_weights = weights[idx]

    cdf = np.cumsum(sort_weights) / np.sum(sort_weights)

    return np.interp(measurement, sort_measurements, cdf, left=0.0, right=1.0)
