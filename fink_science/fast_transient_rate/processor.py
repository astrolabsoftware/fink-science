import numpy as np
import pandas as pd
import os
from pyspark.sql.types import StructType, StructField

from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as F

import fink_science.fast_transient_rate.utils as u
from fink_science.fast_transient_rate import rate_module_output_schema
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
    jdstarthist5sigma = cjd[0]

    for idx in range(len(cfid) - 1):
        if cfid[idx] == fid:
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
    """
    see get_last_alert documentation

    Returns
    -------
    list
        last measurements
    """
    return [
        get_last_alert(fid, cfid, cmagpsf, csigmapsf, cdiffmaglim, cjd)
        for fid, cfid, cmagpsf, csigmapsf, cdiffmaglim, cjd in zip(*args)
    ]


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
    >>> df_concat = concat_col(df_concat, "fid")

    >>> local_df = df_concat.select([
    ... "candidate.fid", "cfid", "cmagpsf", "csigmapsf", "cdiffmaglim", "cjd",
    ... "candidate.jd", "candidate.jdstarthist", "candidate.magpsf", "candidate.sigmapsf"
    ... ]).toPandas()

    >>> type(local_df)
    <class 'pandas.core.frame.DataFrame'>

    >>> fast_transient_rate(local_df, 100, 1)
         jd_first_real_det  jdstarthist_dt  mag_rate  sigma_rate  lower_rate  upper_rate  delta_time  from_upper
    0         2.459511e+06     1158.869884 -0.066805    0.034112   -0.133286   -0.029991   29.892141         1.0
    1         2.459511e+06     1018.095162 -0.102890    0.026445   -0.151063   -0.076359   29.892141         1.0
    2         2.459511e+06     1134.901053  0.018168    0.003675    0.012621    0.023383   29.892141         0.0
    3         2.459511e+06     1137.912095 -0.129177    0.030206   -0.193259   -0.096113   29.892141         1.0
    4         2.459511e+06     1229.753229  0.013787    0.036823   -0.062329    0.050114   29.880023         1.0
    ..                 ...             ...       ...         ...         ...         ...         ...         ...
    315       2.459511e+06     1198.784942 -0.122924    0.032110   -0.181194   -0.090394   29.988160         1.0
    316       2.459513e+06     1209.817639 -0.054985    0.046542   -0.139735   -0.016068   27.934919         1.0
    317       2.459513e+06     1411.094225  0.003907    0.002416   -0.000413    0.007470   22.089444         0.0
    318       2.459513e+06     1209.817639 -0.247882    0.045485   -0.326585   -0.204877   22.093692         1.0
    319       2.459541e+06     1366.055000 -1.769942   24.620849  -41.791666   39.044792    0.010243         0.0
    <BLANKLINE>
    [320 rows x 8 columns]
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
        np.array(
            [
                tmp_last[:, -1],
                jdstarthist_dt,
                res_rate,
                res_sigmarate,
                lower_rate,
                upper_rate,
                dt,
                (~mask_finite_mag) & mask_finite_upper,
            ]
        ).T,
        columns=list(rate_module_output_schema.keys()),
    )


ft_schema = StructType(
    [StructField(k, v, True) for k, v in rate_module_output_schema.items()]
)


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

    Examples
    --------

    """
    pdf = pd.DataFrame(
        {
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
        }
    )

    return fast_transient_rate(pdf, N.values[0], seed.values[0])


def fast_transient_module(spark_df, N, seed=None):
    """
    wrapper function to easily call the fast transient module

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
    >>> df = spark.read.format('parquet').load(ztf_alert_sample)
    >>> df = fast_transient_module(df, 100, 1)
    >>> df.orderBy(["objectId", "candidate.jd"]).select(["objectId", *list(rate_module_output_schema.keys())]).show(5)
    +------------+-----------------+------------------+--------------------+-------------------+--------------------+--------------------+------------------+----------+
    |    objectId|jd_first_real_det|    jdstarthist_dt|            mag_rate|         sigma_rate|          lower_rate|          upper_rate|        delta_time|from_upper|
    +------------+-----------------+------------------+--------------------+-------------------+--------------------+--------------------+------------------+----------+
    |ZTF17aaabbbp|  2459517.8435532| 1425.249652700033|-0.20870715102868334|0.04789120804463472| -0.3064775692990118|-0.16389527305075718| 22.98456020001322|      true|
    |ZTF17aaabbbp|  2459517.8435532| 1425.249652700033|-0.15619291670996355|0.06618118709228582|-0.23448175226201823| -0.1045037986187643|22.959710699971765|      true|
    |ZTF17aaabqqd|  2459512.8257755| 1418.221203699708|-0.00988921883304...|0.00808545546848942|-0.02241833410161...| 0.00395248039266259|28.002858799882233|     false|
    |ZTF17aaabqqd|  2459512.8257755|1419.9723148997873|-0.12165918485605311|0.04948945514809927|-0.21381517662209595|-0.07474257893340712|22.069189800415188|      true|
    |ZTF17aaacfxd|  2459512.8625579|1443.0561111001298|-0.05884086266007449|0.03956762801753035|-0.13966139677545036|-0.01894148628033...| 27.97618049988523|      true|
    +------------+-----------------+------------------+--------------------+-------------------+--------------------+--------------------+------------------+----------+
    only showing top 5 rows
    <BLANKLINE>
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
        cols_before + [df_ft["ft_module"][k].alias(k) for k in rate_module_output_schema.keys()]
    )


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()
    path = os.path.dirname(__file__)
    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
