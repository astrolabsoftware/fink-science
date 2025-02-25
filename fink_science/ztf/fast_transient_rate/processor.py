# Copyright 2020-2025 AstroLab Software
# Author: Roman Le Montagner
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

import numpy as np
import pandas as pd
import os
from pyspark.sql.types import StructType, StructField

from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as F

import fink_science.ztf.fast_transient_rate.utils as u
from fink_science.ztf.fast_transient_rate import rate_module_output_schema
from fink_utils.spark.utils import concat_col

from fink_science.tester import spark_unit_tests
from fink_science import __file__


def get_last_alert(
    fid: int,
    cfid: int,
    cmagpsf: float,
    csigmapsf: float,
    cdiffmaglim: float,
    cjd: float,
) -> list:
    """
    Return the last measurement contains in the alert history used in the fast transient rate

    Parameters
    ----------
    fid : int
        filter id
    cfid : int
        filter id in the history
    cmagpsf : float
        magnitude in the history
    csigmapsf : float
        magnitude error in the history
    cdiffmaglim : float
        upper limit in the history
    cjd : float
        julian date in the history

    Returns
    -------
    list
        list[0]: last magnitude in the history, nan if not available for the current filter
        list[1]: last magnitude error in the history, nan if not available for the current filter
        list[2]: last upper limit in the history, nan if not available for the current filter
        list[3]: last julian date in the history, nan if not available for the current filter
        list[4]: first 5-sigma julian date contains in the history, always available.
    """
    idx_first_mag = np.where(~np.isnan(np.array(cmagpsf, dtype=np.float32)))[0]
    jdstarthist5sigma = cjd[idx_first_mag[0]]

    for idx in range(len(cfid) - 2, -1, -1):
        if cfid[idx] > 2:
            # neither g nor r for ZTF
            # TODO: change the logic for LSST
            return [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                jdstarthist5sigma,
            ]

        elif cfid[idx] == fid:
            if cmagpsf[idx] is None:
                return [
                    float("nan"),
                    float("nan"),
                    cdiffmaglim[idx],
                    cjd[idx],
                    jdstarthist5sigma,
                ]
            else:
                return [
                    cmagpsf[idx],
                    csigmapsf[idx],
                    cdiffmaglim[idx],
                    cjd[idx],
                    jdstarthist5sigma,
                ]

    return [float("nan"), float("nan"), float("nan"), float("nan"), jdstarthist5sigma]


def return_last_alerts(*args) -> list:
    """See get_last_alert documentation

    Returns
    -------
    list
        last measurements
    """
    return [
        get_last_alert(fid, cfid, cmagpsf, csigmapsf, cdiffmaglim, cjd)
        for fid, cfid, cmagpsf, csigmapsf, cdiffmaglim, cjd in zip(*args)
    ]


@profile
def fast_transient_rate(df: pd.DataFrame, N: int, seed: int = None) -> pd.DataFrame:
    """
    Compute the magnitude rate for fast transient detection.

    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing astronomical alerts, must have at least the following columns:
            from the current alert: magpsf, sigmapsf, jd, jdstarthist, fid
            from the alert history: magpsf, sigmapsf, diffmaglim, jd
    N: integer
        Number of values to sample for the error rate computation
    seed: integer
        seed for the random number generator (for reproducibility)

    Return
    ------
    res: pd.DataFrame
        rate estimation and other information
        schema:
            * jd_first_real_det: float
                first variation time at 5 sigma contains in the alert history
            * jdstarthist_dt: float
                delta time between jd_first_real_det and the first variation time at
                    3 sigma (jdstarthist).
                If jdstarthist_dt > 30 days then the first variation time at 5 sigma is false
                    (accurate for fast transient).
            * mag_rate: float
                magnitude rate (mag/day)
            * sigma_rate: float
                magnitude rate error estimation (mag/day)
            * lower_rate: float
                5% percentile of the magnitude rate sampling used for the error computation
            * upper_rate: float
                95% percentile of the magnitude rate sampling used for the error computation
            * delta_time: float
                delta time between the the two measurement used for the magnitude rate
            * from_upper: boolean
                if true, the magnitude rate have been computed with the last upper limit

    Examples
    --------
    >>> spark_df = spark.read.format('parquet').load(ztf_alert_sample)
    >>> df_concat = concat_col(spark_df, "magpsf")
    >>> df_concat = concat_col(df_concat, "sigmapsf")
    >>> df_concat = concat_col(df_concat, "diffmaglim")
    >>> df_concat = concat_col(df_concat, "jd")
    >>> df_concat = concat_col(df_concat, "fid").orderBy(["objectId", "candidate.jd"])

    >>> local_df = df_concat.select([
    ... "candidate.fid", "cfid", "cmagpsf", "csigmapsf", "cdiffmaglim", "cjd",
    ... "candidate.jd", "candidate.jdstarthist", "candidate.magpsf", "candidate.sigmapsf"
    ... ]).toPandas()

    >>> type(local_df)
    <class 'pandas.core.frame.DataFrame'>

    >>> ft_df = fast_transient_rate(local_df, 10000, 2023)
    >>> len(ft_df[ft_df["mag_rate"].abs() > 0.2])
    191
    """
    # create random generator
    rng = np.random.default_rng(seed)

    df = df.reset_index(drop=True)

    # initialize return array
    res_rate = np.empty(len(df), dtype=np.float64)
    res_sigmarate = np.empty(len(df), dtype=np.float64)
    lower_rate = np.empty(len(df), dtype=np.float64)
    upper_rate = np.empty(len(df), dtype=np.float64)

    # fill the empty array with nan values
    res_rate.fill(np.nan)
    res_sigmarate.fill(np.nan)
    lower_rate.fill(np.nan)
    upper_rate.fill(np.nan)

    # get the last measurements for each alerts
    tmp_last = return_last_alerts(
        df["fid"],
        df["cfid"],
        df["cmagpsf"],
        df["csigmapsf"],
        df["cdiffmaglim"],
        df["cjd"],
    )
    tmp_last = np.array(tmp_last)

    # compute delta time between the current alert and the last alert
    dt = df["jd"] - tmp_last[:, -2]

    # compute delta time between the first variation time at 5 sigma in the history
    # and the first variation time at 3 sigma contains in the alert
    # If jdstarthist_dt > 30 days then the first variation time at 5 sigma is false (accurate for fast transient).
    jdstarthist_dt = tmp_last[:, -1] - df["jdstarthist"]

    # detect the real detection
    mask_finite_mag = np.isfinite(tmp_last[:, 0])
    mask_finite_upper = np.isfinite(tmp_last[:, 2])

    # get the idx of the real detections and last upper limit
    idx_last_mag = np.where(mask_finite_mag)[0]
    idx_last_upper = np.where((~mask_finite_mag) & mask_finite_upper)[0]
    idx_valid_data = np.where(mask_finite_mag | mask_finite_upper)[0]

    # sample rate from a gaussian distribution center around the flux with a deviation of error flux
    current_mag_sample = np.empty((N, len(df)), dtype=np.float64)
    current_flux = u.to_flux(df["magpsf"][idx_valid_data])
    current_mag_sample[:, idx_valid_data] = rng.normal(
        current_flux,
        u.to_fluxerr(df["sigmapsf"][idx_valid_data], current_flux),
        (N, len(idx_valid_data)),
    )

    last_flux = u.to_flux(tmp_last[:, 0][idx_last_mag])
    last_mag_sample = rng.normal(
        last_flux,
        u.to_fluxerr(tmp_last[:, 1][idx_last_mag], last_flux),
        (N, len(idx_last_mag)),
    )

    # Fix distribution
    # shift the normal distributions towards positive values
    # and remove 0 values to an epsilon close to 0 but not 0
    # to avoid dividing by 0.

    epsilon_0 = np.finfo(float).eps
    if current_mag_sample[:, idx_valid_data].shape[1] != 0:
        current_mag_sample[:, idx_valid_data] += np.abs(
            np.min(current_mag_sample[:, idx_valid_data])
        )
    current_mag_sample[:, idx_valid_data] = np.where(
        current_mag_sample[:, idx_valid_data] == 0,
        epsilon_0,
        current_mag_sample[:, idx_valid_data],
    )

    if last_mag_sample.shape[1] != 0:
        last_mag_sample += np.abs(np.min(last_mag_sample))
    last_mag_sample = np.where(last_mag_sample == 0, epsilon_0, last_mag_sample)

    # sample upper limit from a uniform distribution starting at 0 until the upper limit
    uniform_upper = rng.uniform(
        0, u.to_flux(tmp_last[:, 2][idx_last_upper]), (N, len(idx_last_upper))
    )

    # compute the rate from the flux difference and convert back to magnitude
    sample_rate = (
        -2.5 * np.log10(current_mag_sample[:, idx_last_mag] / last_mag_sample)
    ).T / u.stack_column(dt[idx_last_mag], N)
    sample_rate_upper = (
        -2.5 * np.log10(current_mag_sample[:, idx_last_upper] / uniform_upper)
    ).T / u.stack_column(dt[idx_last_upper], N)

    # fill the result arrays and return a result dataframe
    res_rate[idx_last_mag] = np.mean(sample_rate, axis=1)
    res_rate[idx_last_upper] = np.mean(sample_rate_upper, axis=1)

    res_sigmarate[idx_last_mag] = np.std(sample_rate, axis=1)
    res_sigmarate[idx_last_upper] = np.std(sample_rate_upper, axis=1)

    lower_rate[idx_last_mag] = np.percentile(sample_rate, 5.0, axis=1)
    lower_rate[idx_last_upper] = np.percentile(sample_rate_upper, 5.0, axis=1)

    upper_rate[idx_last_mag] = np.percentile(sample_rate, 95.0, axis=1)
    upper_rate[idx_last_upper] = np.percentile(sample_rate_upper, 95.0, axis=1)

    return pd.DataFrame(
        np.array([
            tmp_last[:, -1],
            jdstarthist_dt,
            res_rate,
            res_sigmarate,
            lower_rate,
            upper_rate,
            dt,
            (~mask_finite_mag) & mask_finite_upper,
        ]).T,
        columns=list(rate_module_output_schema.keys()),
    )


ft_schema = StructType([
    StructField(k, v, True) for k, v in rate_module_output_schema.items()
])


@pandas_udf(ft_schema)
def magnitude_rate(
    magpsf,
    sigmapsf,
    jd,
    jdstarthist,
    fid,
    cmagpsf,
    csigmapsf,
    cjd,
    cfid,
    cdiffmaglim,
    N,
    seed,
):
    """
    Call the fast_transient_rate within a distributed context (spark pandas udf)

    Parameters
    ----------
    magpsf: pd.Series
        magnitude of the current alert
    sigmapsf: pd.Series
        magnitude error of the current alert
    jd: pd.Series
        julian date of the current alert
    jdstarthist: pd.Series
        first variation time of the alert at 3 sigma
    fid: pd.Series
        filter used during the current exposure
    cmagpsf: pd.Series
        magnitude from the history
    csigmapsf: pd.Series
        magnitude error from the history
    cjd: pd.Series
        julian date of the historical measurements
    cfid: pd.Series
        filter used during the exposure in the history
    cdiffmaglim: pd.Series
        upper limit estimation from the history
    N: pd.Series
        Number of values to sample for the error rate computation
    seed: integer
        seed for the random number generator (for reproducibility)

    Return
    ------
    see fast_transient_rate documentation

    """
    pdf = pd.DataFrame({
        "magpsf": magpsf,
        "sigmapsf": sigmapsf,
        "jd": jd,
        "jdstarthist": jdstarthist,
        "fid": fid,
        "cmagpsf": cmagpsf,
        "csigmapsf": csigmapsf,
        "cdiffmaglim": cdiffmaglim,
        "cjd": cjd,
        "cfid": cfid,
    })

    return fast_transient_rate(pdf, N.to_numpy()[0], seed.to_numpy()[0])


def fast_transient_module(spark_df, N, seed=None):
    """
    Wrapper function to easily call the fast transient module

    Parameters
    ----------
    spark_df : spark dataframe
        dataframe containing alerts
        required columns:
            * candidate.magpsf
            * candidate.sigmapsf
            * candidate.jd
            * candidate.jdstarthist
            * candidate.fid
            * prv_candidate.magpsf
            * prv_candidate.sigmapsf
            * prv_candidate.diffmaglim
            * prv_candidate.jd
            * prv_candidate.fid
    N: integer
        Number of values to sample for the error rate computation
    seed: integer
        seed for the random number generator (for reproducibility)

    Return
    ------
    res: spark dataframe
        same columns than spark_df input + the new columns added by the magnitude_rate function

    Examples
    --------
    >>> from pyspark.sql.functions import abs
    >>> df = spark.read.format('parquet').load(ztf_alert_sample)
    >>> df = fast_transient_module(df, 10000, 2023)
    >>> df.filter(abs(df.mag_rate) > 0.2).count()
    190

    # check robustness for i-band
    >>> df = spark.read.format('parquet').load(ztf_alert_with_i_band)
    >>> df = fast_transient_module(df, 10000, 2023)
    >>> df.filter(abs(df.mag_rate) > 0.2).count()
    119
    """
    cols_before = spark_df.columns

    df_concat = concat_col(spark_df, "magpsf")
    df_concat = concat_col(df_concat, "sigmapsf")
    df_concat = concat_col(df_concat, "diffmaglim")
    df_concat = concat_col(df_concat, "jd")
    df_concat = concat_col(df_concat, "fid")

    df_ft = df_concat.withColumn(
        "ft_module",
        magnitude_rate(
            df_concat["candidate.magpsf"],
            df_concat["candidate.sigmapsf"],
            df_concat["candidate.jd"],
            df_concat["candidate.jdstarthist"],
            df_concat["candidate.fid"],
            df_concat["cmagpsf"],
            df_concat["csigmapsf"],
            df_concat["cjd"],
            df_concat["cfid"],
            df_concat["cdiffmaglim"],
            F.lit(N),
            F.lit(seed),
        ),
    )

    return df_ft.select(
        cols_before
        + [df_ft["ft_module"][k].alias(k) for k in rate_module_output_schema.keys()]
    )


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()
    path = os.path.dirname(__file__)
    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    ztf_alert_with_i_band = (
        "file://{}/data/alerts/20240606_iband_history.parquet".format(path)
    )
    globs["ztf_alert_with_i_band"] = ztf_alert_with_i_band

    # Run the test suite
    spark_unit_tests(globs)
