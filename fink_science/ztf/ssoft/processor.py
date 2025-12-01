# Copyright 2023-2024 AstroLab Software
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
"""file contains scripts and definition for the SSO Fink Table"""

import os
import sys
import time
import datetime

from line_profiler import profile

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import MapType, FloatType, StringType

from fink_utils.sso.spins import estimate_sso_params
from fink_utils.sso.spins import extract_obliquity
from fink_utils.sso.utils import rockify, extract_array_from_series
from fink_utils.sso.utils import compute_light_travel_correction

from fink_science import __file__
from fink_science.tester import spark_unit_tests

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.coordinates import SkyCoord
import astropy.units as u

from asteroid_spinprops.ssolib import modelfit

import logging


_LOG = logging.getLogger(__name__)

COLUMNS = {
    "ssnamenr": {
        "type": "str",
        "description": "Designation (name or number) of the object from MPC archive as given by ZTF",
    },
    "sso_name": {
        "type": "str",
        "description": "Official name or provisional designation of the SSO",
    },
    "sso_number": {"type": "int", "description": "IAU number of the SSO"},
    "last_jd": {
        "type": "double",
        "description": "Julian Date for the last detection in Fink, in UTC",
    },
    "H_1": {
        "type": "double",
        "description": "Absolute magnitude for the ZTF filter band g",
    },
    "H_2": {
        "type": "double",
        "description": "Absolute magnitude for the ZTF filter band r",
    },
    "err_H_1": {
        "type": "double",
        "description": "Uncertainty on the absolute magnitude for the ZTF filter band g",
    },
    "err_H_2": {
        "type": "double",
        "description": "Uncertainty on the absolute magnitude for the ZTF filter band r",
    },
    "min_phase": {
        "type": "double",
        "description": "Minimum phase angle of the observations used to compute the phase function, in degree",
    },
    "min_phase_1": {
        "type": "double",
        "description": "Minimum phase angle of the observations used to compute the phase function for the ZTF filter band g, in degree",
    },
    "min_phase_2": {
        "type": "double",
        "description": "Minimum phase angle of the observations used to compute the phase function for the ZTF filter band r, in degree",
    },
    "max_phase": {
        "type": "double",
        "description": "Maximum phase angle of the observations used to compute the phase function, in degree",
    },
    "max_phase_1": {
        "type": "double",
        "description": "Maximum phase angle of the observations used to compute the phase function for the ZTF filter band g, in degree",
    },
    "max_phase_2": {
        "type": "double",
        "description": "Maximum phase angle of the observations used to compute the phase function for the ZTF filter band r, in degree",
    },
    "chi2red": {"type": "double", "description": "Reduced chi-square of the fit"},
    "rms": {"type": "double", "description": "RMS of the fit, in magnitude"},
    "rms_1": {
        "type": "double",
        "description": "RMS of the fit for the filter band g, in magnitude",
    },
    "rms_2": {
        "type": "double",
        "description": "RMS of the fit for the filter band r, in magnitude",
    },
    "median_error_phot": {
        "type": "double",
        "description": "Median of the 1-sigma uncertainty on the measurements, in magnitude",
    },
    "median_error_phot_1": {
        "type": "double",
        "description": "Median of the 1-sigma uncertainty on the measurements for the filter band g, in magnitude",
    },
    "median_error_phot_2": {
        "type": "double",
        "description": "Median of the 1-sigma uncertainty on the measurements for the filter band r, in magnitude",
    },
    "mean_astrometry": {
        "type": "double",
        "description": "Astrometry: mean of the angular separation between observations and ephemerides, in arcsecond",
    },
    "std_astrometry": {
        "type": "double",
        "description": "Astrometry: standard deviation of the angular separation between observations and ephemerides, in arcsecond",
    },
    "skew_astrometry": {
        "type": "double",
        "description": "Astrometry: skewness of the angular separation between observations and ephemerides",
    },
    "kurt_astrometry": {
        "type": "double",
        "description": "Astrometry: kurtosis of the angular separation between observations and ephemerides",
    },
    "n_obs": {"type": "int", "description": "Number of observations in Fink"},
    "n_obs_1": {
        "type": "int",
        "description": "Number of observations for the ZTF filter band g in Fink",
    },
    "n_obs_2": {
        "type": "int",
        "description": "Number of observations for the ZTF filter band r in Fink",
    },
    "n_days": {
        "type": "int",
        "description": "Number of days between the first and the last observations in Fink",
    },
    "n_days_1": {
        "type": "int",
        "description": "Number of days between the first and the last observations in Fink, for the ZTF filter band g",
    },
    "n_days_2": {
        "type": "int",
        "description": "Number of days between the first and the last observations in Fink, for the ZTF filter band r",
    },
    "fit": {
        "type": "int",
        "description": "Code to assess the quality of the fit: 0: success, 1: bad_vals, 2: MiriadeFail, 3: RunTimError, 4: LinalgError",
    },
    "status": {
        "type": "int",
        "description": "Code for quality `status` (least square convergence): -2: failure, -1 : improper input parameters status returned from MINPACK, 0 : the maximum number of function evaluations is exceeded, 1 : gtol termination condition is satisfied, 2 : ftol termination condition is satisfied, 3 : xtol termination condition is satisfied, 4 : Both ftol and xtol termination conditions are satisfied.",
    },
    "flag": {"type": "int", "description": "TBD"},
    "version": {"type": "str", "description": "Version of the SSOFT YYYY.MM"},
}

COLUMNS_SSHG1G2 = {
    "G1_1": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band g",
    },
    "G1_2": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band r",
    },
    "G2_1": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band g",
    },
    "G2_2": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band r",
    },
    "a_b": {"type": "double", "description": "a/b ratio of the ellipsoid (a>=b>=c)."},
    "a_c": {"type": "double", "description": "a/c ratio of the ellipsoid (a>=b>=c)."},
    "phi0": {
        "type": "double",
        "description": "Initial rotation phase at reference time t0, in radian",
    },
    "alpha0": {
        "type": "double",
        "description": "Right ascension of the spin axis (EQJ2000), in degree",
    },
    "delta0": {
        "type": "double",
        "description": "Declination of the spin axis (EQJ2000), in degree",
    },
    "alpha0_alt": {
        "type": "double",
        "description": "Flipped `alpha0`: (`alpha0` + 180) modulo 360, in degree",
    },
    "delta0_alt": {
        "type": "double",
        "description": "Flipped `delta0`: -`delta0`, in degree",
    },
    "obliquity": {
        "type": "double",
        "description": "Obliquity of the spin axis, in degree",
    },
    "err_G1_1": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band g",
    },
    "err_G1_2": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band r",
    },
    "err_G2_1": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band g",
    },
    "err_G2_2": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band r",
    },
    "err_a_b": {"type": "double", "description": "Uncertainty on a/b"},
    "err_a_c": {"type": "double", "description": "Uncertainty on a/c"},
    "err_phi0": {
        "type": "double",
        "description": "Uncertainty on the initial rotation phase, in radian",
    },
    "err_alpha0": {
        "type": "double",
        "description": "Uncertainty on the right ascension of the spin axis (EQJ2000), in degree",
    },
    "err_delta0": {
        "type": "double",
        "description": "Uncertainty on the declination of the spin axis (EQJ2000), in degree",
    },
    "err_period": {
        "type": "double",
        "description": "Uncertainty on the sidereal period, in hour. Available only from 2024.10",
    },
    "max_cos_lambda": {
        "type": "double",
        "description": "Maximum of the absolute value of the cosine for the aspect angle",
    },
    "mean_cos_lambda": {
        "type": "double",
        "description": "Mean of the absolute value of the cosine for the aspect angle",
    },
    "min_cos_lambda": {
        "type": "double",
        "description": "Minimum of the absolute value of the cosine for the aspect angle",
    },
    "period": {
        "type": "double",
        "description": "Sidereal period estimated, in hour. Available only from 2024.10",
    },
    "period_chi2red": {
        "type": "double",
        "description": "Reduced chi-square for the period estimation. Available only from 2024.10",
    },
}

COLUMNS_SHG1G2 = {
    "G1_1": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band g",
    },
    "G1_2": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band r",
    },
    "G2_1": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band g",
    },
    "G2_2": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band r",
    },
    "R": {"type": "double", "description": "Oblateness of the object"},
    "a_b": {
        "type": "double",
        "description": "a/b ratio of the ellipsoid (a>=b>=c). Estimation based on the fit residuals and the oblateness.",
    },
    "a_c": {
        "type": "double",
        "description": "a/c ratio of the ellipsoid (a>=b>=c). Estimation based on the fit residuals and the oblateness.",
    },
    "alpha0": {
        "type": "double",
        "description": "Right ascension of the spin axis (EQJ2000), in degree",
    },
    "delta0": {
        "type": "double",
        "description": "Declination of the spin axis (EQJ2000), in degree",
    },
    "alpha0_alt": {
        "type": "double",
        "description": "Flipped `alpha0`: (`alpha0` + 180) modulo 360, in degree",
    },
    "delta0_alt": {
        "type": "double",
        "description": "Flipped `delta0`: -`delta0`, in degree",
    },
    "obliquity": {
        "type": "double",
        "description": "Obliquity of the spin axis, in degree",
    },
    "err_G1_1": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band g",
    },
    "err_G1_2": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band r",
    },
    "err_G2_1": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band g",
    },
    "err_G2_2": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band r",
    },
    "err_R": {"type": "double", "description": "Uncertainty on the oblateness"},
    "err_alpha0": {
        "type": "double",
        "description": "Uncertainty on the right ascension of the spin axis (EQJ2000), in degree",
    },
    "err_delta0": {
        "type": "double",
        "description": "Uncertainty on the declination of the spin axis (EQJ2000), in degree",
    },
    "max_cos_lambda": {
        "type": "double",
        "description": "Maximum of the absolute value of the cosine for the aspect angle",
    },
    "mean_cos_lambda": {
        "type": "double",
        "description": "Mean of the absolute value of the cosine for the aspect angle",
    },
    "min_cos_lambda": {
        "type": "double",
        "description": "Minimum of the absolute value of the cosine for the aspect angle",
    },
}

COLUMNS_HG1G2 = {
    "G1_1": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band g",
    },
    "G1_2": {
        "type": "double",
        "description": "G1 phase parameter for the ZTF filter band r",
    },
    "G2_1": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band g",
    },
    "G2_2": {
        "type": "double",
        "description": "G2 phase parameter for the ZTF filter band r",
    },
    "err_G1_1": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band g",
    },
    "err_G1_2": {
        "type": "double",
        "description": "Uncertainty on the G1 phase parameter for the ZTF filter band r",
    },
    "err_G2_1": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band g",
    },
    "err_G2_2": {
        "type": "double",
        "description": "Uncertainty on the G2 phase parameter for the ZTF filter band r",
    },
}

COLUMNS_HG = {
    "G_1": {
        "type": "double",
        "description": "G phase parameter for the ZTF filter band g",
    },
    "G_2": {
        "type": "double",
        "description": "G phase parameter for the ZTF filter band r",
    },
    "err_G_1": {
        "type": "double",
        "description": "Uncertainty on the G phase parameter for the ZTF filter band g",
    },
    "err_G_2": {
        "type": "double",
        "description": "Uncertainty on the G phase parameter for the ZTF filter band r",
    },
}


def estimate_period():
    """TBD

    Should be done only for SSHG1G2

    sb_method: str
        Specify the single-band lomb scargle implementation to use.
        See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargleMultiband.html#astropy.timeseries.LombScargleMultiband.autopower
        If nifty-ls is installed, one can also specify fastnifty. Although
        in this case it does not work yet for Nterms_* higher than 1.
    """
    pass


def extract_ssoft_parameters_sshg1g2():
    """TBD

    Notes
    -----
    For the SSHG1G2 model, the strategy is the following:
    1. Compute parameters as if it was SHG2G1 model (incl. period estimation)
    2. Using previously computed parameters, compute parameters from SSHG1G2
    """
    pass


@pandas_udf(MapType(StringType(), FloatType()), PandasUDFType.SCALAR)
@profile
def extract_ssoft_parameters(
    ssnamenr,
    magpsf,
    sigmapsf,
    jd,
    fid,
    raobs,
    decobs,
    raephem,
    decephem,
    phase,
    dobs,
    dhelio,
    method,
    model,
):
    """Extract phase and spin parameters from Fink alert data using Apache Spark

    Notes
    -----
    Only works for HG, HG1G2, and SHG1G2. Rotation period
    is not estimated here. For SSHG1G2, see <TBD>

    Parameters
    ----------
    ssnamenr: str
        SSO name from ZTF alert packet
    magpsf: float
        Magnitude from ZTF
    sigmapsf: float
        Error estimate on magnitude
    jd: double
        Time of exposition (UTC)
    fid: int
        Filter ID (1=g, 2=r)
    raobs: double
        Observation RAs
    decobs: double
        Observation declinations
    phase:
    dobs:
    dhelio:
    method: str
        Method to compute ephemerides: `ephemcc` or `rest`.
        Use only the former on the Spark Cluster (local installation of ephemcc),
        otherwise use `rest` to call the ssodnet web service.
    model: str
        Model name. Available: HG, HG1G2, SHG1G2, SOCCA


    Returns
    -------
    out: pd.Series
        Series with dictionaries. Keys are parameter names (H, G, etc.)
        depending on the model chosen.
    """
    MODELS = {
        "HG": {"p0": [15.0, 0.15], "bounds": ([-3, 0], [30, 1])},
        "HG1G2": {"p0": [15.0, 0.15, 0.15], "bounds": ([-3, 0, 0], [30, 1, 1])},
        "SHG1G2": {
            "p0": [15.0, 0.15, 0.15, 0.8, np.pi, 0.0],
            "bounds": (
                [-3, 0, 0, 3e-1, 0, -np.pi / 2],
                [30, 1, 1, 1, 2 * np.pi, np.pi / 2],
            ),
        },
        "SOCCA": {
            "p0": [15.0, 0.15, 0.15, np.pi, 0.0, 5.0, 1.05, 1.05, 0.0],
            "bounds": (
                [-3, 0, 0, 0, -np.pi / 2, 2.2 / 24.0, 1, 1, -np.pi / 2],
                [30, 1, 1, 2 * np.pi, np.pi / 2, 1000, 5, 5, np.pi / 2],
            ),
        },
    }

    model_name = model.to_numpy()[0]

    assert model_name in MODELS.keys(), "{} is not supported. Choose among: {}".format(
        model_name, str(MODELS.keys())
    )

    # loop over SSO
    out = []
    for index, ssname in enumerate(ssnamenr.to_numpy()):
        # Construct the dataframe
        magpsf_red = extract_array_from_series(magpsf, index, float) - 5 * np.log10(
            extract_array_from_series(dobs, index, float)
            * extract_array_from_series(dhelio, index, float)
        )
        if model_name == "SOCCA":
            jd_lt = compute_light_travel_correction(
                extract_array_from_series(jd, index, float),
                extract_array_from_series(dobs, index, float),
            )
            pdf = pd.DataFrame({
                "cmred": magpsf_red,
                "csigmapsf": extract_array_from_series(sigmapsf, index, float),
                "Phase": extract_array_from_series(phase, index, float),
                "cfid": extract_array_from_series(fid, index, int),
                "ra": extract_array_from_series(raobs, index, float),
                "dec": extract_array_from_series(decobs, index, float),
                "cjd": jd_lt,
            })
            pdf = pdf.sort_values("cjd")

            # Wrap columns inplace
            pdf = pd.DataFrame({
                colname: [pdf[colname].to_numpy()] for colname in pdf.columns
            })

            # parameter estimation
            outdic = modelfit.get_fit_params(
                pdf,
                flavor=model_name,
                shg1g2_constrained=True,
                blind_scan=True,
                alt_spin=False,
                period_in=None,
                terminator=False,
            )
        else:
            pdf = pd.DataFrame({
                "i:ssnamenr": [ssname] * len(raobs.to_numpy()[index]),
                "i:magpsf": extract_array_from_series(magpsf, index, float),
                "i:sigmapsf": extract_array_from_series(sigmapsf, index, float),
                "i:jd": extract_array_from_series(jd, index, float),
                "i:fid": extract_array_from_series(fid, index, int),
                "i:ra": extract_array_from_series(raobs, index, float),
                "i:dec": extract_array_from_series(decobs, index, float),
                "i:raephem": extract_array_from_series(raephem, index, float),
                "i:decephem": extract_array_from_series(decephem, index, float),
                "i:magpsf_red": magpsf_red,
                "Phase": extract_array_from_series(phase, index, float),
                "Dobs": extract_array_from_series(dobs, index, float),
            })

            pdf = pdf.sort_values("i:jd")

            outdic = estimate_sso_params(
                pdf["i:magpsf_red"].to_numpy(),
                pdf["i:sigmapsf"].to_numpy(),
                np.deg2rad(pdf["Phase"].to_numpy()),
                pdf["i:fid"].to_numpy(),
                np.deg2rad(pdf["i:ra"].to_numpy()),
                np.deg2rad(pdf["i:dec"].to_numpy()),
                jd=pdf["i:jd"].to_numpy(),
                p0=MODELS[model_name]["p0"],
                bounds=MODELS[model_name]["bounds"],
                model=model_name,
                normalise_to_V=False,
            )

        # Add astrometry
        fink_coord = SkyCoord(
            ra=pdf["i:ra"].to_numpy() * u.deg, dec=pdf["i:dec"].to_numpy() * u.deg
        )
        ephem_coord = SkyCoord(
            ra=pdf["i:raephem"].to_numpy() * u.deg,
            dec=pdf["i:decephem"].to_numpy() * u.deg,
        )

        separation = fink_coord.separation(ephem_coord).arcsecond

        outdic["mean_astrometry"] = np.mean(separation)
        outdic["std_astrometry"] = np.std(separation)
        outdic["skew_astrometry"] = skew(separation)
        outdic["kurt_astrometry"] = kurtosis(separation)

        # Time lapse
        outdic["n_days"] = pdf["i:jd"].max() - pdf["i:jd"].min()
        ufilters = np.unique(pdf["i:fid"].to_numpy())
        for filt in ufilters:
            mask = pdf["i:fid"].to_numpy() == filt
            outdic["n_days_{}".format(filt)] = (
                pdf["i:jd"][mask].max() - pdf["i:jd"][mask].min()
            )

        outdic["last_jd"] = pdf["i:jd"].max()

        out.append(outdic)
    return pd.Series(out)


def build_the_ssoft(
    aggregated_filename,
    nparts=400,
    nmin=50,
    frac=None,
    model="SHG1G2",
    version=None,
    sb_method="auto",
    ephem_method="ephemcc",
) -> pd.DataFrame:
    """Build the Fink Flat Table from scratch

    Parameters
    ----------
    aggregated_filename: str
        Aggregated data on HDFS.
    nparts: int, optional
        Number of Spark partitions to used. Default is 400.
        Rule of thumb is nparts = 4 * ncores
    nmin: int, optional
        Minimal number of measurements to select objects (all filters). Default is 50.
    frac: float, optional
        If specified, sample a fraction of the dataset (between 0 and 1). Default is None.
    model: str, optional
        Model name among HG, HG1G2, SHG1G2. Default is SHG1G2.
    version: str, optional
        Version number of the table. By default YYYY.MM.
    ephem_method: str
        Method to compute ephemerides: `ephemcc` (default), or `rest`.

    Returns
    -------
    pdf: pd.DataFrame
        Pandas DataFrame with all the SSOFT data.

    Examples
    --------
    >>> ssoft_hg = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nparts=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='HG',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_hg) == 2, ssoft_hg
    >>> assert "G_1" in ssoft_hg.columns

    >>> col_ssoft_hg = sorted(ssoft_hg.columns)
    >>> expected_cols = sorted({**COLUMNS, **COLUMNS_HG}.keys())
    >>> assert col_ssoft_hg == expected_cols, (col_ssoft_hg, expected_cols)

    >>> ssoft_hg1g2 = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nparts=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='HG1G2',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_hg1g2) == 2, ssoft_hg12
    >>> assert "G1_1" in ssoft_hg1g2.columns

    >>> col_ssoft_hg1g2 = sorted(ssoft_hg1g2.columns)
    >>> expected_cols = sorted({**COLUMNS, **COLUMNS_HG1G2}.keys())
    >>> assert col_ssoft_hg1g2 == expected_cols, (col_ssoft_hg1g2, expected_cols)

    >>> ssoft_shg1g2 = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nparts=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='SHG1G2',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_shg1g2) == 2, ssoft_shg1g2
    >>> assert "R" in ssoft_shg1g2.columns
    >>> assert "a_b" in ssoft_shg1g2.columns

    >>> col_ssoft_shg1g2 = sorted(ssoft_shg1g2.columns)
    >>> expected_cols = sorted({**COLUMNS, **COLUMNS_SHG1G2}.keys())
    >>> assert col_ssoft_shg1g2 == expected_cols, (col_ssoft_shg1g2, expected_cols)

    >>> ssoft_socca = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nparts=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='SOCCA',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_socca) == 2, ssoft_socca
    >>> assert "period" in ssoft_shg1g2.columns
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if version is None:
        now = datetime.datetime.now()
        version = "{}{:02d}".format(now.year, now.month)

    _LOG.info("Reading {} ephemerides".format(aggregated_filename))
    df_ztf = spark.read.format("parquet").load(aggregated_filename)

    _LOG.info("{:,} SSO objects in Fink".format(df_ztf.count()))

    df = df_ztf.withColumn("finkmeasurements", F.size(df_ztf["cra"])).filter(
        F.col("finkmeasurements") >= nmin
    )

    _LOG.info(
        "{:,} SSO objects with more than {} measurements".format(df.count(), nmin)
    )

    # Note: we compute the size of Phase
    # because Phase can be null due to no ephemerides
    df = (
        df.withColumn("ephemmeasurements", F.size(df["Phase"]))
        .filter(F.col("ephemmeasurements") >= nmin)
        .filter(F.size("cmagpsf") == F.size("Phase"))
        .repartition(nparts)
        .cache()
    )

    _LOG.info(
        "{:,} SSO objects with more than {} measurements and ephemerides".format(
            df.count(), nmin
        )
    )

    if frac is not None:
        if frac >= 1:
            _LOG.warning("`frac` should be between 0 and 1.")
            sys.exit()
        df = df.sample(fraction=frac, seed=0).cache()
        _LOG.info(
            "SAMPLE: {:,} SSO objects with more than {} measurements".format(
                df.count(), nmin
            )
        )

    cols = ["ssnamenr", "params"]
    t0 = time.time()
    pdf = (
        df.withColumn(
            "params",
            extract_ssoft_parameters(
                F.col("ssnamenr").astype("string"),
                "cmagpsf",
                "csigmapsf",
                "cjd",
                "cfid",
                "cra",
                "cdec",
                "RA",
                "DEC",
                "Phase",
                "Dobs",
                "Dhelio",
                F.lit(ephem_method),
                F.lit(model),
            ),
        )
        .select(cols)
        .toPandas()
    )

    _LOG.info("Time to extract parameters: {:.2f} seconds".format(time.time() - t0))

    pdf = pd.concat([pdf, pd.json_normalize(pdf.params)], axis=1).drop("params", axis=1)

    sso_name, sso_number = rockify(pdf.ssnamenr.copy())
    pdf["sso_name"] = sso_name
    pdf["sso_number"] = sso_number

    if model == "SHG1G2":
        # compute obliquity
        pdf["obliquity"] = extract_obliquity(
            pdf.sso_name,
            pdf.alpha0,
            pdf.delta0,
        )

        # add flipped spins
        pdf["alpha0_alt"] = (pdf["alpha0"] + 180) % 360
        pdf["delta0_alt"] = -pdf["delta0"]

    pdf["version"] = version

    pdf["flag"] = 0

    return pdf


if __name__ == "__main__":
    """
    """
    globs = globals()
    path = os.path.dirname(__file__)

    aggregated_filename = (
        "file://{}/data/alerts/sso_ztf_lc_aggregated_202504_three_obj.parquet".format(
            path
        )
    )
    globs["aggregated_filename"] = aggregated_filename

    # Run the test suite
    spark_unit_tests(globs)
