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

import os
import io
import json
import requests
import time
import logging
import datetime

from pathlib import Path
from confluent_kafka import Consumer, OFFSET_BEGINNING
from fastavro import schemaless_reader, parse_schema

import numpy as np
import pandas as pd

from astropy.time import Time

_LOG = logging.getLogger(__name__)

# ============================
# Download of FLaapLUC alerts
# ============================


def _reset_to_beginning(consumer: Consumer, partitions: list) -> None:
    """Reset the offset of each queried Kafka partition.

    Reset the commit offset of each queried Kafka partition of FLaapLUC Kafka.
    Used to ensure that all the FLaapLUC stream is queried everytime.

    Parameters
    ----------
    consumer : Consumer
        Kafka Consumer instance to query the FLaapLUC producer.
    partitions : list
        List of Kafka partitions on which to reset the offset.
    """
    for p in partitions:
        p.offset = OFFSET_BEGINNING
    consumer.assign(partitions)


def _FLaapLUC_download(flaapluc_static_path: str) -> pd.DataFrame:
    """Download FLaapLUC alerts.

    Retrieve the FLaapLUC alerts available in the FLaapLUC Kafka stream
    (see Lenain, J.-P., Astron. Comput. 2018, 22).
    Compute the time of the latest detected alert per source and the sigma deviation.

    Parameters
    ----------
    flaapluc_static_path : string
        Path of the .json file containing the globals used for FLaapLUC access.

    Returns
    -------
    dataset : pd.DataFrame
        Pandas DataFrame containg all the objects with an available alert in FLaapLUC
        Kafka. Contains the columns ``4FGL_name``, ``mjd``, ``sigma_deviation``.
    """

    dir_path = Path(os.path.abspath(__file__)).parent
    file_path = dir_path / flaapluc_static_path
    with open(file_path) as f:
        static = json.load(f)

    schema_path = dir_path / static["flaapluc_avro_schema_path"]
    with open(schema_path) as f:
        schema = json.load(f)
        parsed_schema = parse_schema(schema)

    consumer = Consumer(static["flaapluc_kafka_config"])
    consumer.subscribe(
        static["flaapluc_kafka_topics"], on_assign=_reset_to_beginning
    )

    empty_polls = 0
    max_empty = 4

    source_names = []
    times = []
    deviations = []

    while True:
        msg = consumer.poll(5)

        if msg is None:
            empty_polls += 1
            if empty_polls >= max_empty:
                break
            continue

        empty_polls = 0

        if msg.error():
            print("Kafka error:", msg.error())
            continue

        decoded = schemaless_reader(
            io.BytesIO(msg.value()), parsed_schema
        )

        source_names.append(decoded["alert"]["fermi_counterpart_name"])
        times.append(Time(decoded["alert"]["time_last_photon"]).mjd)
        sigma_conversion = (
            decoded["alert"]["flux_threshold"]
            - decoded["alert"]["flux_long_term_average"]
        ) / decoded["alert"]["n_sigma"]
        absolute_deviation = (
            decoded["alert"]["flux"]
            - decoded["alert"]["flux_long_term_average"]
        )
        deviations.append(absolute_deviation / sigma_conversion)

    dataset = pd.DataFrame(
        {
            "4FGL_name": source_names,
            "mjd": times,
            "sigma_deviation": deviations,
            "gamma_flux": decoded["alert"]["flux"]
        }
    )
    return dataset


def catalog_update(
    CTAO_blazar: pd.DataFrame, flaapluc_static_path: str,
    deltatime_check_history: float = 7.
) -> pd.DataFrame:
    """Update catalog with FLaapLUC alerts.

    Update the CTAO-blazar static catalog with FLaapLUC alerts if a flare has been
    detected by Fermi-LAT (see Lenain, J.-P., Astron. Comput. 2018, 22).
    Add a column with the number of sigma (number of standard deviation) deviation
    from the gamma-ray long-time average.
    If no flare has been detected, use Not a Number element instead.

    Parameters
    ----------
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``4FGL_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.
    flaapluc_static_path : string
        Path of the .json file containing the globals used for FLaapLUC access.
    deltatime_check_history : float, optional
        Period (in days) to consider when retrieving the latest FLaapLUC alerts.
        Default: 7.

    Returns
    -------
    updated_CTAO_blazar : pd.DataFrame
        Updated pandas DataFrame of the monitored sources, with an additional
        ``sigma_deviation`` column containing the results from FLaapLUC
    """
    dataset = _FLaapLUC_download(flaapluc_static_path)
    dataset = dataset.drop_duplicates(subset=["4FGL_name"], keep="last")
    dataset = dataset.loc[
        dataset["mjd"] > Time(
            datetime.datetime.now(),
            format="datetime"
        ).mjd - deltatime_check_history
    ]
    updated_CTAO_blazar = CTAO_blazar.join(
        dataset[["4FGL_name", "sigma_deviation", "gamma_flux"]].set_index("4FGL_name"),
        on="4FGL_name"
    )
    return updated_CTAO_blazar


def get_FLaapLUC_deviation(pdf, CTAO_blazar):
    """Retrieve the deviation computed from FLaapLUC.

    Retrieve the number of sigma deviation computed from the FLaapLUC alerts.
    Return ``-1.0`` by default (no FLaapLUC data or the source is not in the catalog).

    Parameters
    ----------
    pdf : pd.DataFrame
        Pandas DataFrame of the alert history containing at least:
        ``candid``, ``ojbectId``, ``cdistnr``, ``cmagpsf``, ``csigmapsf``,
        ``cmagnr``, ``csigmagnr``, ``cisdiffpos``, ``cfid``, ``cjd``,
        ``cstd_flux``, ``csigma_std_flux``.
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``4FGL_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``, ``sigma_deviation``.

    Returns
    -------
    res : float
        Number of sigma (number of standard deviation) deviation
        from the gamma-ray long-time average.
        ``-1,0`` if no flare has been found in FLaapLUC.

    """
    row = CTAO_blazar.loc[CTAO_blazar["ZTF_name"] == pdf["objectId"].iloc[0]]

    res = row["sigma_deviation"].iloc[0]
    if not row.empty and np.isfinite(res) and (res >= 0):
        return res, row["gamma_flux"].iloc[0]
    return -1., -1.


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
        ``Source_name``, ``4FGL_name``, ``ZTF_name``, ``medians``,
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

    Returns the sliding mean over ``integration_period`` days of the standardised flux
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
        ``Source_name``, ``4FGL_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.
    state_key : string
        Key for the threshold to retrieve from ``CTAO_blazar``.
        Either ``low_threshold`` or ``high_threshold``.
    integration_period : float
        Integration period for the computation of the fluence in the
        robustness criterion.

    Returns
    -------
    out: np.float64
        Ratio of the sliding mean over ``integration_period`` days of the standardised
        flux over the precomputed threshold
    """
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
        if hasattr(np, "trapezoid"):
            integral = np.trapezoid(flux[maskNan], x=mtime)
        else:
            integral = np.trapz(flux[maskNan], x=mtime)
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
        ``Source_name``, ``4FGL_name``, ``ZTF_name``, ``medians``,
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
    pdf = pdf[(pdf["mjd"] >= 58000) & np.isin(pdf["filtercode"], [1, 2])].copy()
    pdf = pdf.sort_values("mjd", ascending=True, ignore_index=True)

    return pdf


# ===================================
# Standardisation of the light curve
# ===================================


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


def standardise_dr_lc(
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
        ``Source_name``, ``4FGL_name``, ``ZTF_name``, ``medians``,
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
        median = sub_catalog["medians"].iloc[0][str(filt)]
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
