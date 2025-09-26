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
import os
import contextlib
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
    >>> true_flux = np.array([[1.00000000e+07, 1.00000000e+03], [1.09647820e+11, 1.00000000e+11]])
    >>> true_err = np.array([[9.21034343e+04, 9.21034685e-01], [1.00989370e+10, 9.21034000e+08]])
    >>> np.testing.assert_allclose(np.array([new['cflux'][k] for k in range(2)]), true_flux, rtol=1e-3)
    >>> np.testing.assert_allclose(np.array([new['csigflux'][k] for k in range(2)]), true_err, rtol=1e-3)
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


def remove_nan(pdf):
    """Remove nan/None values from light curves.

    Parameters
    ----------
    pdf: pd.DataFrame
        Must at leat include cflux, based
        on which it will remove Nan/None from the columns:
        'cjd','cmagpsf','csigmapsf','cfid','csigflux','cflux'

    Returns
    -------
    pd.DataFrame
        Original DataFrame with nan/None removed

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
        Include at least cjd, cfid, cflux, csigflux columns.
    salt_model: Model
        Salt model to fit to the light curve.

    Returns
    -------
    list
        List of statistical features
        [amplitude, kurtosis, max_slope, skew]
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
    return list(result)


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
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     sdf = concat_col(sdf, colname, prefix=prefix)

    >>> pdf = sdf.toPandas()
    >>> pdf['distnr'] = pdf['candidate'].apply(lambda x: x[22])
    >>> pdf = compute_flux(pdf)
    >>> pdf = remove_nan(pdf)

    # Fit a light curve
    >>> lc = pdf.iloc[0].copy()

    # Rainbow
    >>> rainbow_model = RainbowFit.from_angstrom(kern.band_wave_aa, with_baseline=False,
    ... temperature=kern.temperature, bolometric=kern.bolometric)
    >>> rainbow_features = fit_rainbow(lc, rainbow_model)

    # Statistical
    >>> stat_features = statistical_features(lc)

    # Salt
    >>> salt_model = quiet_model()
    >>> salt_features = quiet_fit_salt(lc, salt_model)

    # Check their values
    >>> np.testing.assert_allclose(stat_features,
    ... [1.724827e+03, 1.082316e+00, 3.898716e+04, -5.994491e-01],rtol=1e-3)
    >>> np.testing.assert_allclose(salt_features,[2.750825e-01,
    ... 1.232026e+01,   4.719657e-02,   5.983153e+00,
    ... -4.167890e-02, 6.210453e+01],rtol=5e-2)
    >>> np.testing.assert_allclose(rainbow_features,[  1.695213e+01,   6.116788e+04,
    ... 7.864212e+01,   4.913569e+01,
    ... 8.569830e+03,   9.043603e+03,   6.207734e+00,   6.221343e-01,
    ... 2.553051e+01,   1.212390e+00,   9.019693e-01,   9.229657e+00,
    ... 1.846910e+01,   5.537009e-01,   9.581437e-02], rtol=5e-2)

    # Check full feature extraction function
    >>> pdf_check = pdf.copy()
    >>> full_features = extract_features(pdf_check)

    # No alerts should be fitted as they are all <30 days
    >>> np.testing.assert_equal(
    ... np.array(np.sum(full_features.isnull(), axis=1)),
    ... np.array([25]*len(pdf)))
    """
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
            "duration",
            "flux_amplitude",
            "kurtosis",
            "max_slope",
            "skew",
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
            <= np.array([sum(lc["cfid"] == band) for band in np.unique(lc["cfid"])])
        )
        enough_total_points = len(lc["cjd"]) > kern.min_points_total
        duration = np.ptp(lc["cjd"])
        enough_duration = duration > kern.min_duration
        distnr = lc["distnr"]

        if all_valid_bands & enough_total_points & enough_duration:
            rainbow_features = fit_rainbow(lc, rainbow_model)
            salt_features = fit_salt(lc, salt_model)
            stat_features = statistical_features(lc)

            row = [distnr, duration] + stat_features + rainbow_features + salt_features
            pdf.loc[pdf_idx] = row

        else:
            pdf.loc[pdf_idx] = [distnr, duration] + [np.nan] * (np.shape(pdf)[1] - 2)

    return pdf


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/datatest/part-00003-bdab8e46-89c4-4ac1-8603-facd71833e8a-c000.snappy.parquet".format(
        path
    )
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
