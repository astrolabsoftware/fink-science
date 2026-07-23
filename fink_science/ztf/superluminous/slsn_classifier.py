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
from dust_extinction.parameter_averages import F99
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import os
import contextlib
import requests
import urllib.parse
from fink_science import __file__
import io

import warnings
from light_curve.light_curve_py import warnings as rainbow_warnings
warnings.filterwarnings("ignore", category=rainbow_warnings.ExperimentalWarning)


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
        lambda x: np.transpose(
            [mag2fluxcal_snana(*i) for i in zip(x["cmagpsf"], x["csigmapsf"])]
        ),
        axis=1,
    )

    pdf.loc[:, "cflux"] = conversion.apply(lambda x: x[0])
    pdf.loc[:, "csigflux"] = conversion.apply(lambda x: x[1])

    return pdf


def ntrend_changes(cflux, csigflux, cfid, k=3):
    """Return the mean (per band) number of times a light curve abruptly changes trend.

    A trend change is a sign flip between two consecutive flux differences
    (e.g. the light curve was rising and abruptly starts declining), counted
    only when both differences are significant at the `k` sigma level. This
    is meant to flag noisy or bogus-looking light curves, which real SLSNe
    (slow, smooth rise and decline) should not exhibit.

    Parameters
    ----------
    cflux: array
        Flux values of the light curve (single source, all bands mixed).
    csigflux: array
        Uncertainty on `cflux`.
    cfid: array
        Filter/band identifier associated to each point of `cflux`.
    k: float
        Number of sigma above which a flux difference between two
        consecutive points is considered significant. Default is 3.

    Returns
    -------
    float
        Number of trend changes, averaged over the bands present in `cfid`.

    Examples
    --------
    # Two significant trend changes: up, down, up
    >>> cflux = np.array([10., 50., 10., 60., 5.])
    >>> csigflux = np.array([1., 1., 1., 1., 1.])
    >>> cfid = np.array([1, 1, 1, 1, 1])
    >>> ntrend_changes(cflux, csigflux, cfid)
    3.0

    # Monotonic light curve: no trend change
    >>> ntrend_changes(np.array([10., 20., 30., 40., 50.]), csigflux, cfid)
    0.0
    """
    n = []
    for band in np.unique(cfid):
        mask = (cfid == band) & ~np.isnan(cflux)
        x = cflux[mask]
        err = np.array(csigflux[mask], dtype=float)

        dx = np.diff(x)
        sig = np.sqrt(err[:-1] ** 2 + err[1:] ** 2)
        valid = np.abs(dx) > k * sig
        n_turns = np.sum(np.diff(np.sign(dx[valid])) != 0)

        n.append(n_turns)

    return np.mean(n)


def compute_milky_way_extinction(ebv, lambda_angstrom, Rv=3.1):
    """Compute the milky way extinction

    Parameters
    ----------
    ebv: float
        E(B-V) extinction.
    lambda_angstrom: float
        Effective wavelength of the telescope filter expressed in Angstrom
    Rv: float
        Parameter describing the shape of the extinction curve.
        Rv = 3.1 is a standard value in many cases.

    Returns
    -------
    float
        Milky Way extinction A(lambda), in magnitudes, at the requested
        wavelength.

    Examples
    --------
    >>> round(compute_milky_way_extinction(0.5, 6000), 2)
    1.34
    """
    # Filter effective wavelength
    lambda_eff = lambda_angstrom * u.AA

    # Extinction law
    ext = F99(Rv=Rv)

    # Total extinction
    R_lambda = ext(lambda_eff) * Rv
    A_lambda = R_lambda * ebv

    return A_lambda


def abs_peak(app_peak, lambda_angstrom, z, zerr, ebv):
    """Compute the peak absolute magnitude based on redshift, assuming a cosmology

    Notes
    -----
    Uses the redshift uncertainty to return [M(z-zerr), M(z), M(z+zerr)],
    i.e. the absolute magnitude and its two 1-sigma bounds. At fixed
    apparent magnitude, a higher redshift means a larger luminosity
    distance, so the source must be intrinsically *brighter* (more
    negative M) to appear equally bright: M(z+zerr) is therefore the
    brightest bound and M(z-zerr) the faintest one. A flat LambdaCDM
    cosmology (H0=67.8, Om0=0.308) and Milky Way extinction (F99 law) are
    applied. Among the passbands given in `app_peak`, only the one giving
    the brightest (most negative) M(z) is returned.

    Parameters
    ----------
    app_peak: list or float
        Apparent peak magnitude(s), one per passband (or a single float
        for a single passband).
    lambda_angstrom: list or float
        Effective wavelength(s) associated to `app_peak`, expressed in
        Angstrom, in the same order.
    z: float
        Redshift. Returns [nan, nan, nan] if `z` is NaN.
    zerr: float
        Uncertainty on the redshift. Returns [nan, nan, nan] if `zerr`
        is NaN.
    ebv: float
        E(B-V) Milky Way extinction. Negative values are clipped to 0.

    Returns
    -------
    np.array
        Array of 3 floats [M(z-zerr), M(z), M(z+zerr)] for the passband
        giving the brightest absolute magnitude, or [nan, nan, nan] if
        `z` or `zerr` is NaN.

    Examples
    --------
    >>> np.testing.assert_allclose(abs_peak(19, 4000, 0.2, 0.05, 0.1),
    ... [-20.92638971, -21.66227902, -22.25186059], rtol=1e-6)
    >>> np.testing.assert_allclose(abs_peak(19, 4000, 0.2, 0.05, -1),
    ... [-20.48512533, -21.22101463, -21.81059621], rtol=1e-6)
    >>> np.testing.assert_allclose(abs_peak(19, 4000, 0.2, np.nan, 0.1),
    ... [np.nan, np.nan, np.nan], equal_nan=True)
    >>> np.testing.assert_allclose(abs_peak(19, 4000, np.nan, 0.05, 0.1),
    ... [np.nan, np.nan, np.nan], equal_nan=True)
    >>> np.testing.assert_allclose(abs_peak([18, 18], [4400, 6600], 0.12, 0.01, 0.5),
    ... [-22.74727368, -22.96008329, -23.15747603], rtol=1e-6)
    """
    # In case the user gives a single value instead of a list
    app_peak_is_num = (type(app_peak) is float) | (type(app_peak) is int)
    lambda_angstrom_is_num = (type(lambda_angstrom) is float) | (
        type(lambda_angstrom) is int
    )

    if app_peak_is_num & lambda_angstrom_is_num:
        app_peak = [app_peak]
        lambda_angstrom = [lambda_angstrom]

    # In case a negative E(B-V) value is provided
    if ebv < 0:
        ebv = 0

    if (z == z) and (zerr == zerr):
        cosmo = LambdaCDM(H0=67.8, Om0=0.308, Ode0=0.692)

        Ms_lambda = []

        for band in range(len(app_peak)):
            Ms = []
            for k in [-1, 0, 1]:
                effective_z = max(z + k * zerr, 1e-3)
                D_L = cosmo.luminosity_distance(effective_z).to("pc").value
                M = (
                    app_peak[band]
                    - 5 * np.log10(D_L / 10)
                    - 2.5 * np.log10(1 + effective_z)
                    - compute_milky_way_extinction(ebv, lambda_angstrom[band])
                )
                Ms.append(M)
            Ms_lambda.append(Ms)

        # Find the band with the highest absolute magnitude
        brightest = np.argmin(np.array(Ms_lambda)[:, 1])

        return np.array(Ms_lambda[brightest])

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

    # A location within the main SDSS footprint, with a wide radius,
    # should return a real match (DR16 is a frozen data release, so the
    # values are stable over time).
    >>> np.testing.assert_allclose(get_sdss_photoz(180.0, 30.0, radius=60.0),
    ... [0.504143, 0.109665], rtol=1e-4)
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
    >>> np.testing.assert_allclose(pdf["photoz"].values, [np.nan, np.nan], equal_nan=True)
    >>> np.testing.assert_allclose(pdf["photozerr"].values, [np.nan, np.nan], equal_nan=True)

    # Empty input: no SDSS query is made, columns are added empty.
    >>> empty = pd.DataFrame(data={"objectId": [], "ra": [], "dec": []})
    >>> empty = add_all_photoz(empty)
    >>> list(empty.columns)
    ['objectId', 'ra', 'dec', 'photoz', 'photozerr']
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
    >>> np.testing.assert_allclose(get_ebv(np.array([90, 90, 90]), np.array([90, 70, 110])),
    ... [0.25480431, 0.10597386, -1.], rtol=1e-6)
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
        Must at least include objectId, ra, dec columns

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additionnal ebv column.

    Examples
    --------
    >>> pdf = pd.DataFrame(data={"objectId":["a", "b", "a"], "ra": [90, 90, 90], "dec": [70, 90, 70]})
    >>> pdf = add_all_ebv(pdf)
    >>> np.testing.assert_allclose(pdf["ebv"].values,
    ... [0.10597386, 0.25480431, 0.10597386], rtol=1e-6)
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
                lambda row: np.array(
                    [
                        a
                        for a, b in zip(
                            row[k],
                            (np.array(row["cflux"]) == row["cflux"])
                            & (np.array(row["cflux"]) != None),  # noqa: E711
                        )
                        if b
                    ]
                ),
                axis=1,
            )

    return pdf


def remove_bad_bands(pdf):
    """Keep only the g and r bands

    Parameters
    ----------
    pdf: pd.DataFrame
        Must at least include cfid, based
        on which it will remove unwanted bands from the columns:
        "cjd","cmagpsf","csigmapsf","cfid","csigflux","cflux"

    Returns
    -------
    pd.DataFrame
        Original DataFrame with nan/None removed.

    Examples
    --------
    >>> pdf = pd.DataFrame(data={"cflux":[[10, 20, 30, 40]],"cfid":[[1, 2, 3, 3]]})
    >>> result = remove_bad_bands(pdf)
    >>> expected = pd.DataFrame(data={"cflux":[[10, 20]],"cfid":[[1, 2]]})
    >>> pd.testing.assert_frame_equal(result, expected)
    """
    for k in ["cjd", "cmagpsf", "csigmapsf", "csigflux", "cflux", "cfid"]:
        if k in pdf.columns:
            pdf.loc[:, k] = pdf.apply(
                lambda row: np.array(
                    [
                        a
                        for a, b in zip(
                            row[k],
                            (np.isin(row["cfid"], list(kern.band_wave_aa.keys()))),  # noqa: E711
                        )
                        if b
                    ]
                ),
                axis=1,
            )

    return pdf


def fit_rainbow(lc, rainbow_model):
    """Perform a rainbow fit (Russeil et al. 2024) on a light curve.

    Notes
    -----
    The parameter names and their order are given by `rainbow_model.names`
    (e.g. reference_time, amplitude, rise_time, fall_time, Tmin, Tmax,
    t_color for the sigmoid/bazin configuration used by
    `kernel.temperature`/`kernel.bolometric`).
    This function mutates `lc` in place: `cjd`, `cflux`, `csigflux` and
    `cfid` are re-assigned, time-shifted so that `cjd=0` is at peak flux,
    and sorted by increasing time. Pass a copy of the row if the original
    light curve must be preserved (see `extract_features`).

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
        Concatenation of the optimized rainbow parameters, their
        associated uncertainties (parameter / sigma from the iminuit fit),
        and the reduced chi square of the fit -- i.e. `2 * n_params + 1`
        values. Returns a list of NaN of the same length if the fit fails.

    Examples
    --------
    >>> t = np.linspace(0, 60, 20)
    >>> flux_g = 5000. * np.exp(-(t - 15.) / 25.) / (1 + np.exp(-(t - 15.) / 5.))
    >>> flux_r = 0.8 * flux_g
    >>> lc = pd.Series({
    ...     "cjd": np.concatenate([t, t]),
    ...     "cflux": np.concatenate([flux_g, flux_r]),
    ...     "csigflux": np.abs(np.concatenate([flux_g, flux_r])) * 0.05 + 5,
    ...     "cfid": np.array([1] * len(t) + [2] * len(t)),
    ... })
    >>> rainbow_model = RainbowFit.from_angstrom(kern.band_wave_aa, with_baseline=False,
    ... temperature=kern.temperature, bolometric=kern.bolometric)
    >>> rainbow_model.names
    ['reference_time', 'amplitude', 'rise_time', 'fall_time', 'Tmin', 'Tmax', 't_color']
    >>> result = fit_rainbow(lc, rainbow_model)
    >>> len(result) == 2 * len(rainbow_model.names) + 1
    True
    >>> np.testing.assert_allclose(result,
    ... [-7.104360e+00,   7.903653e+03,   6.251021e+00,   2.498412e+01,
    ... 1.372882e+04,   1.384991e+04,   1.524100e+02,  -1.721473e+01,
    ... 1.700575e+01,   3.247533e+01,   2.971197e+01,   9.275860e+00,
    ... 8.686928e+00,   3.149714e-01,   1.381510e-03], rtol=5e-2)
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

    try:
        result, errors = rainbow_model._eval_and_get_errors(
            t=lc["cjd"],
            m=lc["cflux"],
            sigma=lc["csigflux"],
            band=lc["cfid"],
            debug=True,
        )

        return list(result[:-1]) + list(result[:-1] / errors) + [result[-1]]

    except (TypeError, RuntimeError):
        return [np.nan] * (2 * len(rainbow_model.names) + 1)


def fit_salt(lc, salt_model):
    """Perform a salt2 fit (from sncosmo) on a light curve.

    Notes
    -----
    Only ZTF g, r and i bands are supported (`cfid` in {1, 2, 3}). Unlike
    `fit_rainbow`, this function does not mutate `lc`. Redshift is
    constrained to [0, 0.5] during the fit. Use `quiet_fit_salt` instead
    in doctests, to silence sncosmo's model-download messages.

    Parameters
    ----------
    lc: pd.Series
        Include at least cjd, cfid, cflux, csigflux columns.
    salt_model: sncosmo.Model
        Salt2 model to fit to the light curve, e.g.
        `sncosmo.Model(source="salt2")`.

    Returns
    -------
    list
        Concatenation of the optimized salt2 parameters
        (`salt_model.param_names`, typically z, t0, x0, x1, c) and the
        chi square from the fit -- i.e. `n_params + 1` values. Returns a
        list of 6 NaN if the fit fails.

    Examples
    --------
    >>> t = np.linspace(0, 60, 20)
    >>> flux_g = 5000. * np.exp(-(t - 15.) / 25.) / (1 + np.exp(-(t - 15.) / 5.))
    >>> flux_r = 0.8 * flux_g
    >>> lc = pd.Series({
    ...     "cjd": np.concatenate([t, t]),
    ...     "cflux": np.concatenate([flux_g, flux_r]),
    ...     "csigflux": np.abs(np.concatenate([flux_g, flux_r])) * 0.05 + 5,
    ...     "cfid": np.array([1] * len(t) + [2] * len(t)),
    ... })
    >>> salt_model = quiet_model()
    >>> salt_model.param_names
    ['z', 't0', 'x0', 'x1', 'c']
    >>> result = quiet_fit_salt(lc, salt_model)
    >>> len(result) == len(salt_model.param_names) + 1
    True
    >>> np.testing.assert_allclose(result,
    ... [1.682185e-01,  -9.046814e-01,   3.962725e-03,
    ... 4.178744e+00,  -2.626279e-01,   1.202086e+03], rtol=5e-2)
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
    """Compute a few useful statistical features from the light curve package.

    Notes
    -----
    https://github.com/light-curve/light-curve-python

    Parameters
    ----------
    lc: pd.Series
        Include at least cjd, cfid, cmagpsf, cflux, csigflux columns.

    Returns
    -------
    list
        List of 10 statistical features, in this order:
        [amplitude, kurtosis, max_slope, skew, peak_mag_g, peak_mag_r,
        std_flux, q15, q85, ntrends]. `amplitude`, `kurtosis`, `max_slope`
        and `skew` are computed on the flux by the light-curve package.
        `peak_mag_g`/`peak_mag_r` are the brightest (minimum) magnitude
        observed in each band (99 if the band is absent). `std_flux` is
        the standard deviation of the flux normalized by its maximum.
        `q15`/`q85` are the 15th/85th percentile of the time axis,
        shifted so that it starts at 0. `ntrends` is the output of
        `ntrend_changes`.

    Examples
    --------
    >>> t = np.linspace(0, 60, 20)
    >>> flux_g = 5000. * np.exp(-(t - 15.) / 25.) / (1 + np.exp(-(t - 15.) / 5.))
    >>> flux_r = 0.8 * flux_g
    >>> cflux = np.concatenate([flux_g, flux_r])
    >>> lc = pd.Series({
    ...     "cjd": np.concatenate([t, t]),
    ...     "cflux": cflux,
    ...     "csigflux": np.abs(cflux) * 0.05 + 5,
    ...     "cfid": np.array([1] * len(t) + [2] * len(t)),
    ...     "cmagpsf": -2.5 * np.log10(cflux) + 25.,
    ... })
    >>> result = statistical_features(lc)
    >>> len(result)
    10
    >>> np.testing.assert_allclose(result,
    ... [1.342740e+03,  -1.030542e+00,   1.766728e+02,
    ... 2.421149e-01,   1.629598e+01,   1.653826e+01,
    ... 2.475832e-01,   9.000000e+00,   5.100000e+01,
    ... 0.000000e+00], rtol=1e-3)
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

    peak_mag_g = np.min(lc["cmagpsf"][lc["cfid"] == 1], initial=99)
    peak_mag_r = np.min(lc["cmagpsf"][lc["cfid"] == 2], initial=99)

    std = np.std(normed_flux)
    q15 = np.quantile(shifted_time, 0.15)
    q85 = np.quantile(shifted_time, 0.85)
    ntrends = ntrend_changes(lc['cflux'], lc['csigflux'], lc['cfid'])

    return list(result) + [peak_mag_g, peak_mag_r, std, q15, q85, ntrends]


def quiet_model():
    """Build the salt2 sncosmo model, muting its model-download messages.

    Notes
    -----
    The first call to `sncosmo.Model(source="salt2")` on a machine
    downloads and prints information about the salt2 model files, which
    pollutes doctest output. Intended for doctests and any other context
    where this noise is undesirable; use `sncosmo.Model(source="salt2")`
    directly otherwise.

    Returns
    -------
    sncosmo.Model
        A salt2 model, ready to be passed to `fit_salt`.

    Examples
    --------
    >>> quiet_model().source.name
    'salt2'
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return sncosmo.Model(source="salt2")


def quiet_fit_salt(lc, model):
    """Call `fit_salt`, muting sncosmo's model-download messages.

    Notes
    -----
    See `quiet_model`. Intended for doctests.

    Parameters
    ----------
    lc: pd.Series
        Same as `fit_salt`.
    model: sncosmo.Model
        Same as `fit_salt`.

    Returns
    -------
    list
        Same as `fit_salt`.
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
    >>> np.testing.assert_allclose(stat_features,[  8.307904e+02,
    ... 4.843807e-02,   7.573933e+03,  -7.161292e-01,
    ... 1.875300e+01,   1.882850e+01,   1.383518e-01,
    ... 9.992026e+00,   2.499306e+01,   0.000000e+00], rtol=1e-3)
    >>> np.testing.assert_allclose(salt_features,[  1.374512e-01,
    ... -1.201602e+01,   3.522748e-03,   9.219506e+00,
    ... 3.321469e-02,   4.337947e+01], rtol=5e-2)
    >>> np.testing.assert_allclose(rainbow_features,
    ... [ -2.046804e+00,   4.928837e+03,   2.208002e+01,   2.879719e+01,
    ... 9.048897e+03,   9.814914e+03,   1.417986e+00,  -4.941401e-01,
    ... 1.016117e+01,   6.211122e+00,   1.094480e+00,   7.419188e+00,
    ... 1.500786e+01,   1.129233e+00,   1.211680e-01], rtol=5e-2)

    # Check full feature extraction function
    >>> pdf_check = pdf.copy()
    >>> full_features = extract_features(pdf_check)

    # Only the fake alert should pass the cuts
    >>> np.testing.assert_equal(
    ... np.array(np.sum(full_features.iloc[-30:].isnull(), axis=1)),
    ... np.array([ 0, 31, 31,  0,  0, 31,  0,  0, 31, 31, 31, 31,  0,  0,
    ... 0,  0, 31, 0, 31,  0, 31,  0,  0,  0,  0, 31, 31,  0, 31,  0]))

    >>> list(full_features.columns) == ["distnr", "ra", "dec", "ebv", "duration",
    ... "flux_amplitude", "kurtosis", "max_slope", "skew", "peak_mag_g", "peak_mag_r",
    ... "std_flux", "q15", "q85", "ntrends", "reference_time", "amplitude", "rise_time", "fall_time",
    ... "Tmin", "Tmax", "t_color", "snr_reference_time", "snr_amplitude", "snr_rise_time",
    ... "snr_fall_time", "snr_Tmin", "snr_Tmax", "snr_t_color", "chi2_rainbow", "z", "t0",
    ... "x0", "x1", "c", "chi2_salt"]
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
            "peak_mag_g",
            "peak_mag_r",
            "std_flux",
            "q15",
            "q85",
            "ntrends",
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
