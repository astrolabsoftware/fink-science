# Copyright 2025 AstroLab Software
# Author: Etienne Russeil
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

import pandas as pd
import numpy as np
from light_curve.light_curve_py import RainbowFit
import sncosmo
from astropy.table import Table
import light_curve as lcpckg
import fink_science.ztf.superluminous.kernel as kern
from fink_science.tester import spark_unit_tests
from fink_utils.photometry.conversion import mag2fluxcal_snana
import astropy.units as u
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import os
import contextlib
import requests
import urllib.parse
from fink_science import __file__
import io


def compute_flux(pdf):
    """Convert cmagpsf and csigmapsf to cflux and csigflux.

    Notes
    -----
    Add two columns to the original dataset

    Parameters
    ----------
    pdf: pd.DataFrame
        Include at least cmagpsf and csigmapsf columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two extra columns
        cflux and csigflux

    Examples
    --------
    >>> pdf = pd.DataFrame(data=
    ...   {"cmagpsf":[[10, 20], [-0.1, 0]],
    ...    "csigmapsf":[[0.01, 0.001], [0.1, 0.01]]})
    >>> new = compute_flux(pdf)
    >>> type(new) == type(pd.DataFrame())
    True
    >>> true_flux = np.array([[1.00000000e+07, 1.00000000e+03], [1.09647820e+11, 1.00000000e+11]])
    >>> true_err = np.array([[9.21034343e+04, 9.21034685e-01], [1.00989370e+10, 9.21034000e+08]])
    >>> np.testing.assert_allclose(np.array([new["cflux"][k] for k in range(2)]), true_flux, rtol=1e-3)
    >>> np.testing.assert_allclose(np.array([new["csigflux"][k] for k in range(2)]), true_err, rtol=1e-3)
    """
    conversion = pdf[["cmagpsf", "csigmapsf"]].apply(
        lambda x: np.transpose([
            mag2fluxcal_snana(*i) for i in zip(x["cmagpsf"], x["csigmapsf"])
        ]),
        axis=1,
    )

    pdf.loc[:, "cflux"] = conversion.apply(lambda x: x[0])
    pdf.loc[:, "csigflux"] = conversion.apply(lambda x: x[1])

    return pdf


def abs_peak(app_peak, z, zerr, ebv):
    """Compute the peak absolute magnitude based on redshift, assuming a cosmology

    Notes
    -----
    Uses uncertainty to return [M(z+zerr), M(z), M(z-zerr)]

    Parameters
    ----------
    app_peak: float
        Apparent peak magnitude.
    z: float
        Redshift
    zerr: float
        Uncertainty on the redshift
    ebv: float
        E(B-V) extinction

    Examples
    --------
    >>> abs_peak(19, 0.2, 0.05, 0.1)
    array([-20.49163613, -21.1351084 , -21.63604614])
    >>> abs_peak(19, 0.2, 0.05, -1)
    array([-20.18163613, -20.8251084 , -21.32604614])
    >>> abs_peak(19, 0.2, np.nan, 0.1)
    array([ nan,  nan,  nan])
    >>> abs_peak(19, np.nan, 0.05, 0.1)
    array([ nan,  nan,  nan])
    """
    if ebv < 0:
        ebv = 0

    if (z == z) and (zerr == zerr):
        cosmo = LambdaCDM(H0=67.8, Om0=0.308, Ode0=0.692)
        Rv = 3.1

        Ms = []
        for k in [-1, 0, 1]:
            effective_z = max(z + k * zerr, 1e-3)
            D_L = cosmo.luminosity_distance(effective_z).to("pc").value
            M = (
                app_peak
                - 5 * np.log10(D_L / 10)
                + 2.5 * np.log10(1 + effective_z)
                - Rv * ebv
            )
            Ms.append(M)

        return np.array(Ms)

    return np.array([np.nan, np.nan, np.nan])


def get_sdss_photoz(ra, dec, radius=0.2):
    """Retrieve photoz from SDSS

    Parameters
    ----------
    ra: array
        Right ascension of the source(s).
    dec: array
        Declination of the source(s).
    radius: float
        Maximum angular distance for association
        with SDSS candidate.
        Default is 0.2

    Returns
    -------
    tuple
        Photometric redshift and it"s uncertainty

    Examples
    --------
    # We cannot check for a precise location in case SDSS servers are not responding
    # After 5 sec, it will time out and output np.nan
    >>> get_sdss_photoz(66, 66)
    (nan, nan)
    """
    try:
        query = f"""
        SELECT TOP 1 p.objID, p.ra, p.dec, z.z AS photoz, z.zErr AS photozErr
        FROM PhotoObj AS p
        JOIN Photoz AS z ON p.objID = z.objID
        JOIN dbo.fGetNearbyObjEq({ra}, {dec}, {radius}) AS n
          ON p.objID = n.objID
        ORDER BY n.distance
        """

        base_url = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
        params = {"cmd": query, "format": "json"}

        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        response = requests.get(url, timeout=5)

        # check we get a valid response
        if response.status_code != 200:
            return np.nan, np.nan

        payload = response.json()

        # check the payload is not empty
        if isinstance(payload, list) and len(payload) > 0:
            table = payload[0].get("Rows", [])
        else:
            return np.nan, np.nan

        if len(table) > 0:
            return table[0]["photoz"], table[0]["photozErr"]

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return np.nan, np.nan
    return np.nan, np.nan


def add_all_photoz(pdf):
    """Add the photo-z and uncertainty columns to a dataframe.

    Parameters
    ----------
    pdf: pd.DataFrame
        Must at leat include objectId, ra, dec columns

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additionnal
        photo-z and uncertainty columns.

    Examples
    --------
    # We cannot check for a precise location in case SDSS servers are not responding
    # After 5 sec, it will time out and output np.nan
    >>> pdf = pd.DataFrame(data={"objectId":["a", "b"],
    ... "ra": [66, 66], "dec": [66, 66]})
    >>> pdf = add_all_photoz(pdf)
    >>> pdf["photoz"].values
    array([ nan,  nan])
    >>> pdf["photozerr"].values
    array([ nan,  nan])
    """
    if len(pdf) > 0:
        unique_objs = pdf.drop_duplicates(subset="objectId")[["objectId", "ra", "dec"]]
        unique_objs[["photoz", "photozerr"]] = unique_objs.apply(
            lambda x: get_sdss_photoz(x["ra"], x["dec"]), axis=1, result_type="expand"
        )
        pdf = pdf.merge(
            unique_objs[["objectId", "photoz", "photozerr"]], on="objectId", how="left"
        )

    else:
        pdf["photoz"] = []
        pdf["photozerr"] = []

    return pdf


def get_ebv(ra, dec):
    """Retrieve E(B-V) extinction based on coordinates

    Parameters
    ----------
    ra: array
        Right ascension of the source(s).
    dec: array
        Declination of the source(s).

    Returns
    -------
    array
        E(B-V) extinction of the source(s).

    Examples
    --------
    >>> get_ebv(np.array([90, 90, 90]), np.array([90, 70, 110]))
    array([ 0.25480431,  0.10597386, -1.        ])
    """
    result = -np.ones(len(dec))
    valid_mask = np.abs(dec) <= 90
    sfd = SFDQuery()
    coord = SkyCoord(ra=ra[valid_mask] * u.deg, dec=dec[valid_mask] * u.deg)
    ebv = sfd(coord)
    result[valid_mask] = ebv
    return result


def add_all_ebv(pdf):
    """Add the E(B-V) column to a dataframe.

    Parameters
    ----------
    pdf: pd.DataFrame
        Must at leat include objectId, ra, dec columns

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additionnal ebv column.

    Examples
    --------
    >>> pdf = pd.DataFrame(data={"objectId":["a", "b", "a"], "ra": [90, 90, 90], "dec": [70, 90, 70]})
    >>> pdf = add_all_ebv(pdf)
    >>> pdf["ebv"].values
    array([ 0.10597386,  0.25480431,  0.10597386])
    """
    unique_objs = pdf.drop_duplicates(subset="objectId")[["objectId", "ra", "dec"]]
    unique_objs["ebv"] = get_ebv(unique_objs["ra"].values, unique_objs["dec"].values)
    pdf = pdf.merge(unique_objs[["objectId", "ebv"]], on="objectId", how="left")
    return pdf


def remove_nan(pdf):
    """Remove nan/None values from light curves.

    Parameters
    ----------
    pdf: pd.DataFrame
        Must at leat include cflux, based
        on which it will remove Nan/None from the columns:
        "cjd","cmagpsf","csigmapsf","cfid","csigflux","cflux"

    Returns
    -------
    pd.DataFrame
        Original DataFrame with nan/None removed.

    Examples
    --------
    >>> pdf = pd.DataFrame(data={"cflux":[[10, 20, np.nan, None]],"cfid":[[1, 2, 1, 2]]})
    >>> result = remove_nan(pdf)
    >>> expected = pd.DataFrame(data={"cflux":[[10, 20]],"cfid":[[1, 2]]})
    >>> pd.testing.assert_frame_equal(result, expected)
    """
    for k in ["cjd", "cmagpsf", "csigmapsf", "cfid", "csigflux", "cflux"]:
        if k in pdf.columns:
            pdf.loc[:, k] = pdf.apply(
                lambda row: np.array([
                    a
                    for a, b in zip(
                        row[k],
                        (np.array(row["cflux"]) == row["cflux"])
                        & (np.array(row["cflux"]) != None),  # noqa: E711
                    )
                    if b
                ]),
                axis=1,
            )

    return pdf


def fit_rainbow(lc, rainbow_model):
    """Perform a rainbow fit (Russeil et al. 2024) on a light curve.

    Parameters
    ----------
    lc: pd.Series
        Include at least cjd, cfid, cflux, csigflux columns.
    rainbow_model: RainbowFit
        Rainbow model to fit to the light curve.
        (https://github.com/light-curve/light-curve-python)

    Returns
    -------
    list
        List of optimized rainbow parameters, their
        associated uncertainties from iminuit fit and
        the reduced chi square of the fit.
    """
    # Shift time
    lc["cjd"] = lc["cjd"] - lc["cjd"][np.argmax(lc["cflux"])]

    # Sort values
    zipped = zip(lc["cjd"], lc["cflux"], lc["csigflux"], lc["cfid"])
    lc["cjd"], lc["cflux"], lc["csigflux"], lc["cfid"] = zip(
        *sorted(zipped, key=lambda x: x[0])
    )
    lc["cjd"], lc["cflux"], lc["csigflux"], lc["cfid"] = (
        np.array(lc["cjd"]),
        np.array(lc["cflux"]),
        np.array(lc["csigflux"]),
        np.array(lc["cfid"]),
    )

    # t_scaler = Scaler.from_time(lc["cjd"])
    # m_scaler = MultiBandScaler.from_flux(lc["cflux"], lc["cfid"], with_baseline=False)

    try:
        result, errors = rainbow_model._eval_and_get_errors(
            t=lc["cjd"],
            m=lc["cflux"],
            sigma=lc["csigflux"],
            band=lc["cfid"],
            debug=True,
        )

        return list(result[:-1]) + list(result[:-1] / errors) + [result[-1]]

    except RuntimeError:
        return [np.nan] * (2 * len(rainbow_model.names) + 1)


def fit_salt(lc, salt_model):
    """Perform a salt fit (from sncosmo) on a light curve.

    Parameters
    ----------
    lc: pd.Series
        Include at least cjd, cfid, cflux, csigflux columns.
    salt_model: Model
        Salt model to fit to the light curve.

    Returns
    -------
    list
        List of optimized salt parameters along with
        the chi square from the fit.
    """
    int_to_filter = {1: "ztfg", 2: "ztfr", 3: "ztfi"}
    lc_table = Table(
        data={
            "time": lc["cjd"] - lc["cjd"][np.argmax(lc["cflux"])],
            "band": [int_to_filter[k] for k in lc["cfid"]],
            "flux": lc["cflux"],
            "fluxerr": lc["csigflux"],
            "zp": [25.0] * len(lc["cjd"]),
            "zpsys": ["ab"] * len(lc["cjd"]),
        }
    )

    try:
        # run the fit
        result, fitted_model = sncosmo.fit_lc(
            lc_table,
            salt_model,
            ["z", "t0", "x0", "x1", "c"],  # parameters of model to vary
            bounds={"z": (0, 0.5)},
        )

        return list(result.parameters) + [result.chisq]

    except RuntimeError:
        return [np.nan] * 6


def statistical_features(lc):
    """Compute few useful statistical features from the light curve package

    Notes
    -----
    https://github.com/light-curve/light-curve-python

    Parameters
    ----------
    lc: pd.Series
        Include at least cjd, cfid, cmagpsf, cflux, csigflux columns.
    salt_model: Model
        Salt model to fit to the light curve.

    Returns
    -------
    list
        List of statistical features
        [amplitude, kurtosis, max_slope, skew, peak_magn, std_flux, q15_time, q85_time]
    """
    amplitude = lcpckg.Amplitude()
    kurtosis = lcpckg.Kurtosis()
    max_slope = lcpckg.MaximumSlope()
    skew = lcpckg.Skew()

    # Feature extractor, it will evaluate all features in more efficient way
    extractor = lcpckg.Extractor(amplitude, kurtosis, max_slope, skew)

    # Array with all 4 extracted features
    result = extractor(
        lc["cjd"],
        lc["cflux"],
        lc["csigflux"],
        sorted=True,
        check=True,
    )

    normed_flux = lc["cflux"] / np.max(lc["cflux"])
    shifted_time = lc["cjd"] - np.min(lc["cjd"])
    peak_mag = np.min(lc["cmagpsf"])
    std = np.std(normed_flux)
    q15 = np.quantile(shifted_time, 0.15)
    q85 = np.quantile(shifted_time, 0.85)
    return list(result) + [peak_mag, std, q15, q85]


def quiet_model():
    """Call the salt model but muting download messages.

    Notes
    -----
    Intended for doctests.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return sncosmo.Model(source="salt2")


def quiet_fit_salt(lc, model):
    """Fit the salt model but muting download messages.

    Notes
    -----
    Intended for doctests.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return fit_salt(lc, model)


def extract_features(data):
    """Extract all features, i.e. Rainbow + salt + some statistical features for a set of light curves.

    Parameters
    ----------
    data: pd.DataFrame
        Each row correspond to a light curve
        Columns are cjd, cfid, cflux, csigflux, distnr.

    Returns
    -------
    pd.DataFrame
        with columns being features and rows indexed
        the same as the input DataFrame

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> sdf = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ["jd", "fid", "magpsf", "sigmapsf"]

    # Use for creating temp name
    >>> prefix = "c"
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     sdf = concat_col(sdf, colname, prefix=prefix)

    >>> pdf = sdf.toPandas()

    # Create a fake light curve that would pass the cuts
    >>> faketime, fakemag = np.linspace(0, 50, 10), np.linspace(18, 15, 10)
    >>> fakesig, fakefid = [0.01] * len(fakemag), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    >>> pdf.loc[[pdf.index[-1]], "cjd"] = pd.Series([np.array(faketime)], index=pdf.index[[-1]])
    >>> pdf.loc[[pdf.index[-1]], "cmagpsf"] = pd.Series([np.array(fakemag)], index=pdf.index[[-1]])
    >>> pdf.loc[[pdf.index[-1]], "csigmapsf"] = pd.Series([np.array(fakesig)], index=pdf.index[[-1]])
    >>> pdf.loc[[pdf.index[-1]], "cfid"] = pd.Series([np.array(fakefid)], index=pdf.index[[-1]])

    >>> pdf["distnr"] = pdf["candidate"].apply(lambda x: x[22])
    >>> pdf["ra"] = pdf["candidate"].apply(lambda x: x[15])
    >>> pdf["dec"] = pdf["candidate"].apply(lambda x: x[16])
    >>> pdf = compute_flux(pdf)
    >>> pdf = remove_nan(pdf)

    # Fit a light curve
    >>> lc = pdf.iloc[0].copy()

    # Rainbow
    >>> rainbow_model = RainbowFit.from_angstrom(kern.band_wave_aa, with_baseline=False,
    ... temperature=kern.temperature, bolometric=kern.bolometric)
    >>> rainbow_features = fit_rainbow(lc, rainbow_model)
    >>> lc['candid']
    3229390645815015015

    # Statistical
    >>> stat_features = statistical_features(lc)

    # Salt
    >>> salt_model = quiet_model()
    >>> salt_features = quiet_fit_salt(lc, salt_model)

    # Check their values
    >>> np.testing.assert_allclose(stat_features,[   8.307904e+02,
    ... 4.843807e-02,   7.573933e+03,  -7.161292e-01,
    ... 1.875300e+01,   1.383518e-01,   9.992026e+00,   2.499306e+01], rtol=1e-3)
    >>> np.testing.assert_allclose(salt_features,[  1.374512e-01,
    ... -1.201602e+01,   3.522748e-03,   9.219506e+00,
    ... 3.321469e-02,   4.337947e+01], rtol=5e-2)
    >>> np.testing.assert_allclose(rainbow_features,
    ... [ -2.161259e+00,   4.886508e+03,   2.196836e+01,   2.740976e+01,
    ... 9.102432e+03,   9.948595e+03,   1.403806e+00,  -5.663001e-01,
    ... 1.050990e+01,   6.421245e+00,   1.106539e+00,   7.157673e+00,
    ... 1.364669e+01,   1.184238e+00,   1.194966e-01], rtol=5e-2)

    # Check full feature extraction function
    >>> pdf_check = pdf.copy()
    >>> full_features = extract_features(pdf_check)

    # Only the fake alert should pass the cuts
    >>> np.testing.assert_equal(
    ... np.array(np.sum(full_features.iloc[-30:].isnull(), axis=1)),
    ... np.array([ 29, 29, 29,  29, 29, 29,  29, 29, 29,  29, 29, 29,  29,
    ... 29, 29, 29,  29, 29, 29,  29, 29, 29, 29, 29, 29,  29, 29, 29, 29, 0]))

    >>> list(full_features.columns) == ["distnr", "ra", "dec", "ebv", "duration",
    ... "flux_amplitude", "kurtosis", "max_slope", "skew", "peak_mag", "std_flux", "q15",
    ... "q85", "reference_time", "amplitude", "rise_time", "fall_time", "Tmin",
    ... "Tmax", "t_color", "snr_reference_time", "snr_amplitude", "snr_rise_time",
    ... "snr_fall_time", "snr_Tmin", "snr_Tmax", "snr_t_color", "chi2_rainbow",
    ... "z", "t0", "x0", "x1", "c", "chi2_salt"]
    True
    """
    data = add_all_ebv(data)

    rainbow_model = RainbowFit.from_angstrom(
        kern.band_wave_aa,
        with_baseline=False,
        temperature=kern.temperature,
        bolometric=kern.bolometric,
    )

    salt_model = sncosmo.Model(source="salt2")

    rainbow_pnames = rainbow_model.names
    salt_pnames = salt_model.param_names

    pdf = pd.DataFrame(
        columns=[
            "distnr",
            "ra",
            "dec",
            "ebv",
            "duration",
            "flux_amplitude",
            "kurtosis",
            "max_slope",
            "skew",
            "peak_mag",
            "std_flux",
            "q15",
            "q85",
        ]
        + rainbow_pnames
        + ["snr_" + k for k in rainbow_pnames]
        + ["chi2_rainbow"]
        + salt_pnames
        + ["chi2_salt"]
    )

    for pdf_idx in range(len(data)):
        lc = data.iloc[pdf_idx].copy()

        all_valid_bands = all(
            kern.min_points_perband
            <= np.array([sum(lc["cfid"] == band) for band in list(kern.band_wave_aa)])
        )
        enough_total_points = len(lc["cjd"]) > kern.min_points_total
        duration = np.ptp(lc["cjd"])
        enough_duration = duration > kern.min_duration
        distnr = lc["distnr"]
        ra = lc["ra"]
        dec = lc["dec"]
        ebv = lc["ebv"]

        if all_valid_bands & enough_total_points & enough_duration:
            rainbow_features = fit_rainbow(lc, rainbow_model)
            salt_features = fit_salt(lc, salt_model)
            stat_features = statistical_features(lc)

            row = (
                [distnr, ra, dec, ebv, duration]
                + stat_features
                + rainbow_features
                + salt_features
            )
            pdf.loc[pdf_idx] = row

        else:
            pdf.loc[pdf_idx] = [distnr, ra, dec, ebv, duration] + [np.nan] * (
                np.shape(pdf)[1] - 5
            )

    return pdf


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/superluminous_test_alerts.parquet".format(
        path
    )
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
