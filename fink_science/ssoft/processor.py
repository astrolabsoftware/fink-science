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
import re
import sys
import time
import datetime

from line_profiler import profile

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import MapType, FloatType, StringType

from fink_utils.sso.utils import query_miriade, query_miriade_epehemcc, get_miriade_data
from fink_utils.sso.utils import compute_light_travel_correction
from fink_utils.sso.spins import estimate_sso_params
from fink_utils.sso.periods import estimate_synodic_period

from fink_science import __file__
from fink_science.tester import spark_unit_tests

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.coordinates import SkyCoord
import astropy.units as u

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
    "period": {
        "type": "double",
        "description": "Sidereal period estimated, in hour. Available only from 2024.10",
    },
    "period_chi2red": {
        "type": "double",
        "description": "Reduced chi-square for the period estimation. Available only from 2024.10",
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


def remove_leading_zeros(val):
    """Iteratively remove leading zeros from a string

    Parameters
    ----------
    val: str
        A string

    Returns
    -------
    The input string with leading zeros removed

    Examples
    --------
    >>> string = '0abcd'
    >>> remove_leading_zeros(string)
    'abcd'

    >>> string = '000000a0bcd'
    >>> remove_leading_zeros(string)
    'a0bcd'

    >>> string = 'toto'
    >>> remove_leading_zeros(string)
    'toto'
    """
    if val.startswith("0"):
        return remove_leading_zeros(val[1:])
    else:
        return val


def process_regex(regex, data):
    """Extract parameters from a regex given the data

    Parameters
    ----------
    regex: str
        Regular expression to use
    data: str
        Data entered by the user

    Returns
    -------
    parameters: dict or None
        Parameters (key: value) extracted from the data
    """
    template = re.compile(regex)
    m = template.match(data)
    if m is None:
        return None

    parameters = m.groupdict()
    return parameters


def angle_between_vectors(v1, v2):
    """Compute the angle between two 3D vectors.

    Parameters
    ----------
    v1 : list or np.ndarray
        The first 3D vector.
    v2 : list or np.ndarray
        The second 3D vector.

    Returns
    -------
    float
        The angle between the two vectors in radians.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Clip to handle numerical issues
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle


@pandas_udf(MapType(StringType(), FloatType()), PandasUDFType.SCALAR)
@profile
def estimate_sso_params_spark(
    ssnamenr, magpsf, sigmapsf, jd, fid, ra, dec, method, model, sb_method, uid
):
    """Extract phase and spin parameters from Fink alert data using Apache Spark

    Notes
    -----
    For the SSHG1G2 model, the strategy is the following:
    1. Compute parameters as if it was SHG2G1 model (incl. period estimation)
    2. Using previously computed parameters, compute parameters from SSHG1G2

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
    ra: double
        RA
    dec: double
        Declination
    method: str
        Method to compute ephemerides: `ephemcc` or `rest`.
        Use only the former on the Spark Cluster (local installation of ephemcc),
        otherwise use `rest` to call the ssodnet web service.
    model: str
        Model name. Available: HG, HG1G2, SHG1G2, SSHG1G2
    sb_method: str
        Specify the single-band lomb scargle implementation to use.
        See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargleMultiband.html#astropy.timeseries.LombScargleMultiband.autopower
        If nifty-ls is installed, one can also specify fastnifty. Although
        in this case it does not work yet for Nterms_* higher than 1.
    uid: int
        Unique ID used internally when writing files on disk by eproc.

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
        "SSHG1G2": {
            "p0": [15.0, 0.15, 0.15, np.pi, 0.0, 5.0, 1.05, 1.05, 0.0],
            "bounds": (
                [-3, 0, 0, 0, -np.pi / 2, 2.2 / 24.0, 1, 1, -np.pi / 2],
                [30, 1, 1, 2 * np.pi, np.pi / 2, 1000, 5, 5, np.pi / 2],
            ),
        },
    }

    # loop over SSO
    out = []
    for index, ssname in enumerate(ssnamenr.to_numpy()):
        uid_object = int(uid.to_numpy()[index] * 1e7)

        # First get ephemerides data
        pdf_sso = pd.DataFrame({
            "i:ssnamenr": [ssname] * len(ra.to_numpy()[index]),
            "i:magpsf": magpsf.to_numpy()[index].astype(float),
            "i:sigmapsf": sigmapsf.to_numpy()[index].astype(float),
            "i:jd": jd.to_numpy()[index].astype(float),
            "i:fid": fid.to_numpy()[index].astype(int),
            "i:ra": ra.to_numpy()[index].astype(float),
            "i:dec": dec.to_numpy()[index].astype(float),
        })

        pdf_sso = pdf_sso.sort_values("i:jd")

        if method.to_numpy()[0] == "ephemcc":
            # hardcoded for the Spark cluster!
            parameters = {
                "outdir": "/tmp/ramdisk/spins",
                "runner_path": "/tmp/fink_run_ephemcc4.sh",
                "userconf": "/tmp/.eproc-4.3",
                "iofile": "/tmp/default-ephemcc-observation.xml",
            }
        elif method.to_numpy()[0] == "rest":
            parameters = {}
        pdf = get_miriade_data(
            pdf_sso,
            observer="I41",
            rplane="1",
            tcoor=5,
            withecl=False,
            method=method.to_numpy()[0],
            parameters=parameters,
            uid=uid_object,
        )

        if "i:magpsf_red" not in pdf.columns:
            out.append({"fit": 2, "status": -2})
        else:
            # TODO: for SSHG1G2, d'abord faire SHG1G2
            if model.to_numpy()[0] in ["SSHG1G2", "SHG1G2"]:
                initial_model = "SHG1G2"
                # Both needs to use SHG1G2
                outdic = estimate_sso_params(
                    pdf["i:magpsf_red"].to_numpy(),
                    pdf["i:sigmapsf"].to_numpy(),
                    np.deg2rad(pdf["Phase"].to_numpy()),
                    pdf["i:fid"].to_numpy(),
                    np.deg2rad(pdf["i:ra"].to_numpy()),
                    np.deg2rad(pdf["i:dec"].to_numpy()),
                    jd=pdf["i:jd"].to_numpy(),
                    p0=MODELS[initial_model]["p0"],
                    bounds=MODELS[initial_model]["bounds"],
                    model=initial_model,
                    normalise_to_V=False,
                )
            else:
                initial_model = model.to_numpy()[0]
                outdic = estimate_sso_params(
                    pdf["i:magpsf_red"].to_numpy(),
                    pdf["i:sigmapsf"].to_numpy(),
                    np.deg2rad(pdf["Phase"].to_numpy()),
                    pdf["i:fid"].to_numpy(),
                    p0=MODELS[initial_model]["p0"],
                    bounds=MODELS[initial_model]["bounds"],
                    model=initial_model,
                    normalise_to_V=False,
                )

            filts = np.unique(pdf["i:fid"].to_numpy())
            is_fit_ok = np.all(["H_{}".format(filt) in outdic for filt in filts])
            if is_fit_ok:
                # Add synodic period estimation
                period, chi2red_period = estimate_synodic_period(
                    pdf=pdf,
                    phyparam=outdic,
                    flavor=initial_model,
                    sb_method=sb_method.to_numpy()[0],
                    Nterms_base=1,
                    Nterms_band=1,
                    period_range=(1.0 / 24.0, 30.0),  # 1h to 1 month
                    lt_correction=True,
                )

                outdic["period"] = period
                outdic["period_chi2red"] = chi2red_period
            else:
                outdic["period"] = np.nan
                outdic["period_chi2red"] = np.nan

            # Full inversion using pre-computed SHG1G2 & period
            if (model.to_numpy()[0] == "SSHG1G2") and ~np.isnan(outdic["period"]):
                # Light travel correction
                jd_lt = compute_light_travel_correction(pdf["i:jd"], pdf["Dobs"])

                # synodic period
                synodic_period_days = outdic["period"] / 24.0

                # compute phase shit
                if method.to_numpy()[0] == "ephemcc":
                    # Get previous ephemerides
                    eph_t = pd.DataFrame({
                        "px": pdf["px"],
                        "py": pdf["py"],
                        "pz": pdf["pz"],
                    })

                    # Get shifted ephemerides
                    eph_tp = query_miriade_epehemcc(
                        ssname,
                        pdf["i:jd"] + synodic_period_days,
                        tcoor=2,
                        parameters=parameters,
                        uid=uid_object,
                    )
                elif method.to_numpy()[0] == "rest":
                    eph_t = query_miriade(ssname, pdf["i:jd"], tcoor=2)
                    eph_tp = query_miriade(
                        ssname, pdf["i:jd"] + synodic_period_days, tcoor=2
                    )
                angle = [
                    angle_between_vectors(
                        eph_t.loc[i, ["px", "py", "pz"]],
                        eph_tp.loc[i, ["px", "py", "pz"]],
                    )
                    for i in range(len(pdf))
                ]
                phase_shift = np.median(angle)

                # loop over 4 cases -- append _ij to outdic
                configurations = {
                    "00": [outdic["alpha0"], outdic["delta0"], phase_shift],
                    "01": [outdic["alpha0"], outdic["delta0"], -phase_shift],
                    "10": [
                        (outdic["alpha0"] + 180) % 360,
                        -outdic["delta0"],
                        phase_shift,
                    ],
                    "11": [
                        (outdic["alpha0"] + 180) % 360,
                        -outdic["delta0"],
                        -phase_shift,
                    ],
                }

                outdic_final = {}
                for key in configurations.keys():
                    # compute sidereal period
                    sidereal_period_days = (
                        synodic_period_days
                        * 2
                        * np.pi
                        / (2 * np.pi + configurations[key][2])
                    )

                    # TODO: extend `estimate_sso_parameters` to take p0 per filter for H & G
                    p0 = [
                        outdic.get("H_1", outdic["H_2"]),
                        outdic.get("G1_1", outdic["G1_2"]),
                        outdic.get("G2_1", outdic["G2_2"]),
                        np.deg2rad(configurations[key][0]),
                        np.deg2rad(configurations[key][1]),
                        sidereal_period_days,
                        outdic["a_b"],
                        outdic["a_c"],
                        0.0,
                    ]

                    # Constrained Fit
                    # in-place replacement of parameters `outdic`
                    outdic_tmp = estimate_sso_params(
                        pdf["i:magpsf_red"].to_numpy(),
                        pdf["i:sigmapsf"].to_numpy(),
                        np.deg2rad(pdf["Phase"].to_numpy()),
                        pdf["i:fid"].to_numpy(),
                        ra=np.deg2rad(pdf["i:ra"].to_numpy()),
                        dec=np.deg2rad(pdf["i:dec"].to_numpy()),
                        jd=jd_lt.to_numpy(),
                        p0=p0,
                        bounds=MODELS["SSHG1G2"]["bounds"],
                        model="SSHG1G2",
                        normalise_to_V=False,
                    )

                    # Only if the fit is successful
                    if "period" in outdic_tmp:
                        # day to hour units
                        outdic_tmp["period"] = 24 * outdic_tmp["period"]

                        # need to repopulate this field from the periodogram estimation
                        outdic_tmp["period_chi2red"] = chi2red_period

                    # rename
                    outdic_tmp = {
                        k + "_{}".format(key): v for k, v in outdic_tmp.items()
                    }

                    # append
                    outdic_final = {**outdic_final, **outdic_tmp}
                outdic_final["phase_shift_deg"] = np.rad2deg(phase_shift)
                outdic = outdic_final

            # Add astrometry
            fink_coord = SkyCoord(
                ra=pdf["i:ra"].to_numpy() * u.deg, dec=pdf["i:dec"].to_numpy() * u.deg
            )
            ephem_coord = SkyCoord(
                ra=pdf["RA"].to_numpy() * u.deg, dec=pdf["Dec"].to_numpy() * u.deg
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


def correct_ztf_mpc_names(ssnamenr):
    """Remove trailing 0 at the end of SSO names from ZTF

    e.g. 2010XY03 should read 2010XY3

    Parameters
    ----------
    ssnamenr: np.array
        Array with SSO names from ZTF

    Returns
    -------
    out: np.array
        Array with corrected names from ZTF

    Examples
    --------
    >>> ssnamenr = np.array(['2010XY03', '2023AB0', '2023XY00', '345', '2023UY12'])
    >>> ssnamenr_alt = correct_ztf_mpc_names(ssnamenr)

    # the first ones changed
    >>> assert ssnamenr_alt[0] == '2010XY3'
    >>> assert ssnamenr_alt[1] == '2023AB'
    >>> assert ssnamenr_alt[2] == '2023XY'

    >>> assert np.all(ssnamenr_alt[3:] == ssnamenr[3:])
    """
    # remove numbered
    regex = "^\d+$"  # noqa: W605
    template = re.compile(regex)
    unnumbered = np.array([template.findall(str(x)) == [] for x in ssnamenr])

    # Extract names
    regex = "(?P<year>\d{4})(?P<letter>\w{2})(?P<end>\d+)$"  # noqa: W605
    processed = [process_regex(regex, x) for x in ssnamenr[unnumbered]]

    def f(x, y):
        """Correct for trailing 0 in SSO names

        Parameters
        ----------
        x: dict, or None
            Data extracted from the regex
        y: str
            Corresponding ssnamenr

        Returns
        -------
        out: str
            Name corrected for trailing 0 at the end (e.g. 2010XY03 should read 2010XY3)
        """
        if x is None:
            return y
        else:
            return "{}{}{}".format(
                x["year"], x["letter"], remove_leading_zeros(x["end"])
            )

    corrected = np.array([f(x, y) for x, y in zip(processed, ssnamenr[unnumbered])])

    ssnamenr[unnumbered] = corrected

    return ssnamenr


def rockify(ssnamenr: pd.Series):
    """Extract names and numbers from ssnamenr

    Parameters
    ----------
    ssnamenr: pd.Series of str
        SSO names as given in ZTF alert packets

    Returns
    -------
    sso_name: np.array of str
        SSO names according to quaero
    sso_number: np.array of int
        SSO numbers according to quaero
    """
    import rocks

    # prune names
    ssnamenr_alt = correct_ztf_mpc_names(ssnamenr.to_numpy())

    # rockify
    names_numbers = rocks.identify(ssnamenr_alt)

    sso_name = np.transpose(names_numbers)[0]
    sso_number = np.transpose(names_numbers)[1]

    return sso_name, sso_number


def angular_separation(lon1, lat1, lon2, lat2):
    """Angular separation between two points on a sphere.

    Notes
    -----
    Stolen from astropy -- for version <5

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : `~astropy.coordinates.Angle`, `~astropy.units.Quantity` or float
        Longitude and latitude of the two points. Quantities should be in
        angular units; floats in radians.

    Returns
    -------
    angular separation : `~astropy.units.Quantity` ['angle'] or float
        Type depends on input; ``Quantity`` in angular units, or float in
        radians.

    Notes
    -----
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
    """
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.arctan2(np.hypot(num1, num2), denominator)


def extract_obliquity(sso_name, alpha0, delta0):
    """Extract obliquity using spin values, and the BFT information

    Parameters
    ----------
    sso_name: np.array or pd.Series of str
        SSO names according to quaero (see `rockify`)
    alpha0: np.array or pd.Series of double
        RA of the pole [degree]
    delta0: np.array or pd.Series of double
        DEC of the pole [degree]

    Returns
    -------
    obliquity: np.array of double
        Obliquity for each object [degree]
    """
    import rocks

    cols = [
        "sso_name",
        "orbital_elements.node_longitude.value",
        "orbital_elements.inclination.value",
    ]
    pdf_bft = rocks.load_bft(columns=cols)

    sub = pdf_bft[cols]

    pdf = pd.DataFrame({"sso_name": sso_name, "alpha0": alpha0, "delta0": delta0})

    pdf = pdf.merge(sub[cols], left_on="sso_name", right_on="sso_name", how="left")

    # Orbit
    lon_orbit = (pdf["orbital_elements.node_longitude.value"] - 90).to_numpy()
    lat_orbit = (90.0 - pdf["orbital_elements.inclination.value"]).to_numpy()

    # Spin -- convert to EC
    ra = np.nan_to_num(pdf.alpha0.to_numpy()) * u.degree
    dec = np.nan_to_num(pdf.delta0.to_numpy()) * u.degree

    # Trick to put the object "far enough"
    coords_spin = SkyCoord(ra=ra, dec=dec, distance=200 * u.parsec, frame="hcrs")

    # in radian
    lon_spin = coords_spin.heliocentricmeanecliptic.lon.value
    lat_spin = coords_spin.heliocentricmeanecliptic.lat.value

    obliquity = np.degrees(
        angular_separation(
            np.radians(lon_spin),
            np.radians(lat_spin),
            np.radians(lon_orbit),
            np.radians(lat_orbit),
        )
    )

    return obliquity


def aggregate_sso_data(output_filename=None):
    """Aggregate all SSO data in Fink

    Data is read from HDFS on VirtualData

    Parameters
    ----------
    output_filename: str, optional
        If given, save data on HDFS. Cannot overwrite. Default is None.

    Returns
    -------
    df_grouped: Spark DataFrame
        Spark DataFrame with aggregated SSO data.
    """
    spark = SparkSession.builder.getOrCreate()
    cols0 = ["candidate.ssnamenr"]
    cols = [
        "candidate.ra",
        "candidate.dec",
        "candidate.magpsf",
        "candidate.sigmapsf",
        "candidate.fid",
        "candidate.jd",
    ]

    df = (
        spark.read.format("parquet")
        .option("basePath", "archive/science")
        .load("archive/science")
    )
    df_agg = (
        df.select(cols0 + cols)
        .filter(F.col("roid") == 3)
        .groupBy("ssnamenr")
        .agg(*[
            F.collect_list(col.split(".")[1]).alias("c" + col.split(".")[1])
            for col in cols
        ])
    )

    if output_filename is not None:
        df_agg.write.parquet(output_filename)

    return df_agg


def build_the_ssoft(
    aggregated_filename=None,
    nproc=80,
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
    aggregated_filename: str, optional
        If given, read aggregated data on HDFS. Default is None.
    nproc: int, optional
        Number of cores to used. Default is 80.
    nmin: int, optional
        Minimal number of measurements to select objects (all filters). Default is 50.
    frac: float, optional
        If specified, sample a fraction of the dataset (between 0 and 1). Default is None.
    model: str, optional
        Model name among HG, HG1G2, SHG1G2. Default is SHG1G2.
    version: str, optional
        Version number of the table. By default YYYY.MM.
    sb_method: str
        Specify the single-band lomb scargle implementation to use.
        See https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargleMultiband.html#astropy.timeseries.LombScargleMultiband.autopower
        If nifty-ls is installed, one can also specify fastnifty. Although
        in this case it does not work yet for Nterms_* higher than 1.
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
    ...     nproc=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='HG',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_hg) == 3, ssoft_hg
    >>> assert "G_1" in ssoft_hg.columns

    >>> ssoft_hg1g2 = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nproc=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='HG1G2',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_hg1g2) == 3, ssoft_hg12
    >>> assert "G1_1" in ssoft_hg1g2.columns

    >>> ssoft_shg1g2 = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nproc=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='SHG1G2',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_shg1g2) == 3, ssoft_shg1g2
    >>> assert "R" in ssoft_shg1g2.columns
    >>> assert "a_b" in ssoft_shg1g2.columns

    >>> ssoft_sshg1g2 = build_the_ssoft(
    ...     aggregated_filename=aggregated_filename,
    ...     nproc=1,
    ...     nmin=50,
    ...     frac=None,
    ...     model='SSHG1G2',
    ...     version=None,
    ...     ephem_method="rest",
    ...     sb_method="fastnifty")
    <BLANKLINE>
    >>> assert len(ssoft_sshg1g2) == 3, ssoft_sshg1g2
    >>> assert "a_b_00" in ssoft_sshg1g2.columns
    >>> assert "a_b_01" in ssoft_sshg1g2.columns
    >>> assert "a_b_10" in ssoft_sshg1g2.columns
    >>> assert "a_b_11" in ssoft_sshg1g2.columns
    >>> assert "obliquity_00" in ssoft_sshg1g2.columns
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    if version is None:
        now = datetime.datetime.now()
        version = "{}.{:02d}".format(now.year, now.month)

    if aggregated_filename is None:
        _LOG.info("Reconstructing SSO data...")
        t0 = time.time()
        aggregated_filename = "sso_aggregated_{}".format(version)
        aggregate_sso_data(output_filename=aggregated_filename)
        _LOG.info(
            "Time to reconstruct SSO data: {:.2f} seconds".format(time.time() - t0)
        )
    df_ztf = spark.read.format("parquet").load(aggregated_filename)

    _LOG.info("{:,} SSO objects in Fink".format(df_ztf.count()))

    df = (
        df_ztf.withColumn("nmeasurements", F.size(df_ztf["cra"]))
        .filter(F.col("nmeasurements") >= nmin)
        .repartition(nproc)
        .cache()
    )

    _LOG.info(
        "{:,} SSO objects with more than {} measurements".format(df.count(), nmin)
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
            estimate_sso_params_spark(
                F.col("ssnamenr").astype("string"),
                "cmagpsf",
                "csigmapsf",
                "cjd",
                "cfid",
                "cra",
                "cdec",
                F.lit(ephem_method),
                F.lit(model),
                F.lit(sb_method),
                F.rand(42),
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

    if model == "SSHG1G2":
        pdf["obliquity_00"] = extract_obliquity(
            pdf.sso_name, pdf.alpha0_00, pdf.delta0_00
        )
        pdf["obliquity_01"] = extract_obliquity(
            pdf.sso_name, pdf.alpha0_01, pdf.delta0_01
        )
        pdf["obliquity_10"] = extract_obliquity(
            pdf.sso_name, pdf.alpha0_10, pdf.delta0_10
        )
        pdf["obliquity_11"] = extract_obliquity(
            pdf.sso_name, pdf.alpha0_11, pdf.delta0_11
        )

    pdf["version"] = version

    pdf["flag"] = 0

    return pdf


if __name__ == "__main__":
    """
    """
    globs = globals()
    path = os.path.dirname(__file__)

    aggregated_filename = (
        "file://{}/data/alerts/sso_aggregated_2024.09_test_sample.parquet".format(path)
    )
    globs["aggregated_filename"] = aggregated_filename

    # Run the test suite
    spark_unit_tests(globs)
