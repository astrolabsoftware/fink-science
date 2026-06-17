# Copyright 2025 AstroLab Software
# Author: Julien Peloton, Etienne Russeil, Daniel Perley
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

from pyspark.sql.functions import (
    col,
    lit,
    coalesce,
    expr,
    when,
    least,
    array_min,
    size,
)
from pyspark.sql.functions import abs as abs_
from line_profiler import profile

from fink_science import __file__
from fink_science.tester import spark_unit_tests


def extract_intermediate_cols(df):
    """Add intermediate columns"""
    df = (
        df.withColumn("m_now", col("candidate.magpsf"))
        .withColumn("t_now", col("candidate.jd"))
        .withColumn("m_app", col("candidate.magap"))
        .withColumn("fid_now", col("candidate.fid"))
        .withColumn("sgscore1", col("candidate.sgscore1"))
        .withColumn("sgscore2", col("candidate.sgscore2"))
        .withColumn("sgscore3", col("candidate.sgscore3"))
        .withColumn("srmag1", col("candidate.srmag1"))
        .withColumn("srmag2", col("candidate.srmag2"))
        .withColumn("srmag3", col("candidate.srmag3"))
        .withColumn("sgmag1", col("candidate.sgmag1"))
        .withColumn("simag1", col("candidate.simag1"))
        .withColumn("szmag1", col("candidate.szmag1"))
        .withColumn("rbscore", col("candidate.rb"))
        .withColumn("magnr", col("candidate.magnr"))
        .withColumn(
            "distnr",
            when(col("candidate.distnr") < 0, 99).otherwise(col("candidate.distnr")),
        )
        .withColumn(
            "distpsnr1",
            when(col("candidate.distpsnr1") < 0, 99).otherwise(
                col("candidate.distpsnr1")
            ),
        )
        .withColumn(
            "distpsnr2",
            when(col("candidate.distpsnr2") < 0, 99).otherwise(
                col("candidate.distpsnr2")
            ),
        )
        .withColumn(
            "distpsnr3",
            when(col("candidate.distpsnr3") < 0, 99).otherwise(
                col("candidate.distpsnr3")
            ),
        )
        .withColumn("scorr", col("candidate.scorr"))
        .withColumn("fwhm", col("candidate.fwhm"))
        .withColumn("elong", col("candidate.elong"))
        .withColumn("nbad", col("candidate.nbad"))
        .withColumn("ra", col("candidate.ra"))
        .withColumn("dec", col("candidate.dec"))
        .withColumn("chipsf", col("candidate.chipsf"))
        .withColumn("ssdistnr", col("candidate.ssdistnr"))
        .withColumn("ssmagnr", col("candidate.ssmagnr"))
        .withColumn("ssnamenr", col("candidate.ssnamenr"))
        .withColumn("t_start", col("candidate.jdstarthist"))
        .withColumn("neargaia", col("candidate.neargaia"))
        .withColumn("maggaia", col("candidate.maggaia"))
        .withColumn("neargaiabright", col("candidate.neargaiabright"))
        .withColumn("maggaiabright", col("candidate.maggaiabright"))
        .withColumn("isdiffpos", col("candidate.isdiffpos"))
        .withColumn("drb", col("candidate.drb"))
        .withColumn("age", col("candidate.jd") - col("candidate.jdstarthist"))
        .withColumn("ndethist", col("candidate.ndethist"))
        .withColumn(
            "ps1mag",
            when(
                (col("candidate.srmag1") > 0) & (col("candidate.srmag1") < 30),
                col("candidate.srmag1"),
            )
            .when(
                (col("candidate.simag1") > 0) & (col("candidate.simag1") < 30),
                col("candidate.simag1"),
            )
            .when(
                (col("candidate.sgmag1") > 0) & (col("candidate.sgmag1") < 30),
                col("candidate.sgmag1"),
            )
            .otherwise(lit(99)),
        )
    )

    srmag1_val = when(col("candidate.srmag1") > 0, col("candidate.srmag1")).otherwise(
        lit(99)
    )
    simag1_val = when(col("candidate.simag1") > 0, col("candidate.simag1")).otherwise(
        lit(99)
    )
    sgmag1_val = when(col("candidate.sgmag1") > 0, col("candidate.sgmag1")).otherwise(
        lit(99)
    )
    szmag1_val = when(col("candidate.szmag1") > 0, col("candidate.szmag1")).otherwise(
        lit(99)
    )

    df = df.withColumn(
        "ps1maxmag", least(srmag1_val, simag1_val, sgmag1_val, szmag1_val)
    )

    df = df.withColumn(
        "prevpasscount",
        size(
            expr("""
        filter(prv_candidates, cand ->
        (candidate.jd - cand.jd < 30)
        AND (candidate.jd - cand.jd > 0.75)
        AND (CAST(cand.isdiffpos AS STRING) IN ('1', 't', 'true'))
        AND (cand.magpsf > 0)
        AND (cand.magpsf < 19.8))""")
        ),
    )

    df = df.withColumn(
        "prevnegcount",
        size(
            expr(
                """filter(prv_candidates, cand -> CAST(cand.isdiffpos AS STRING) IN ('0', 'f', 'false'))"""
            )
        ),
    )

    return df


def is_faint(df, colname="faint"):
    """Is considered as faint"""
    # 1. Filter prv_candidates where fid matches candidate.fid
    filtered_mags = expr("""transform(
            filter(prv_candidates, cand -> cand.fid = candidate.fid),
            cand -> cand.magpsf)""")

    # 2. Minimum of that filtered list (might be empty)
    min_prev_mag = array_min(filtered_mags)

    # 3. Peak mag is the minimum between current magpsf and all previous with same fid
    df = df.withColumn("peakmag", least(col("candidate.magpsf"), min_prev_mag))

    df = df.withColumn(
        colname,
        (col("m_now") >= 19.8)
        | (
            expr("""
            size(filter(prv_candidates, cand ->
                    abs(t_now - cand.jd) < 0.75 AND
                    CAST(isdiffpos AS STRING) IN ('1', 't', 'true') AND
                    cand.magpsf >= 19)) > 0""")
        ),
    )

    return df


def positive_subtraction(df, colname="positivesubtraction"):
    """Is brighter than the template image."""
    df = df.withColumn(colname, col("isdiffpos").cast("string").isin("1", "t", "true"))
    return df


def real_transient(df, colname="real"):
    """Is likely a genuine astrophysical transient and not an artifact."""
    not_real_cond = (
        (col("rbscore") <= 0.2)
        | (
            (col("rbscore") < 0.35)
            & (abs_(col("neargaia")) < 1)
            & (abs_(col("maggaia")) < 17)
        )
        | (
            (col("rbscore") < 0.35)
            & (col("distpsnr1") < 1)
            & (col("sgscore1") > 0.49)
            & (
                (abs_(col("srmag1")) < 17)
                | (abs_(col("simag1")) < 17)
                | (abs_(col("szmag1")) < 16.5)
            )
        )
        | (
            (col("rbscore") < 0.45)
            & (abs_(col("neargaia")) < 1.5)
            & (abs_(col("maggaia")) < 15.5)
        )
        | (
            (col("rbscore") < 0.45)
            & (col("distpsnr1") < 1.5)
            & (col("sgscore1") > 0.49)
            & (
                (abs_(col("srmag1")) < 15.5)
                | (abs_(col("simag1")) < 15.5)
                | (abs_(col("szmag1")) < 15)
            )
        )
        | (col("drb") < 0.8)
        | (
            (col("drb") < 0.9)
            & (col("distpsnr1") < 3)
            & (col("ps1mag") < 16)
            & (col("age") > 90)
        )
        | (
            (col("drb") < 0.9)
            & (col("distpsnr1") < 1.1)
            & (col("ps1mag") < 18)
            & (col("age") > 90)
        )
        | (
            (col("drb") < 0.95)
            & (col("distpsnr1") < 1.5)
            & (col("ps1mag") < 15.5)
            & (col("age") > 90)
        )
        | (
            (col("drb") < 0.95)
            & (col("distpsnr1") < 0.8)
            & (col("ps1mag") < 17.5)
            & (col("age") > 90)
        )
    )

    df = df.withColumn(colname, ~not_real_cond)

    return df


def point_underneath(df, colname="pointunderneath"):
    """Is likely sitting on top of or blended with a star in Pan-STARRS."""
    # First condition: high star-galaxy score, moderate distance
    cond1 = (col("sgscore1") > 0.76) & (col("distpsnr1") < 2.5)

    # Second condition: moderate sgscore1 + small distance + strong red-z color
    cond2 = (
        (col("sgscore1") > 0.2)
        & (col("distpsnr1") < 1)
        & (col("srmag1") > 0)
        & (col("szmag1") > 0)
        & ((col("srmag1") - col("szmag1")) > 3)
    )

    # Third condition: moderate sgscore1 + small distance + strong red-i color
    cond3 = (
        (col("sgscore1") > 0.2)
        & (col("distpsnr1") < 1)
        & (col("srmag1") > 0)
        & (col("simag1") > 0)
        & ((col("srmag1") - col("simag1")) > 3)
    )

    # Combine all
    df = df.withColumn(colname, cond1 | cond2 | cond3)

    return df


def bright_star(df, colname="brightstar"):
    """Is likely contaminated by a nearby bright star."""
    # Condition 1: Close to a bright Gaia star
    cond1 = (abs_(col("neargaiabright")) < 20) & (abs_(col("maggaiabright")) < 12)

    # Condition 2: PS1 source 1, r-band
    cond2 = (
        (col("distpsnr1") < 20) & (abs_(col("srmag1")) < 14) & (col("sgscore1") > 0.9)
    )

    # Condition 3: PS1 source 2, r-band
    cond3 = (
        (col("distpsnr2") < 20) & (abs_(col("srmag2")) < 14) & (col("sgscore2") > 0.9)
    )

    # Condition 4: PS1 source 3, r-band
    cond4 = (
        (col("distpsnr3") < 20) & (abs_(col("srmag3")) < 14) & (col("sgscore3") > 0.9)
    )

    # Condition 5: PS1 source 1, i-band
    cond5 = (
        (col("distpsnr1") < 20) & (abs_(col("simag1")) < 14) & (col("sgscore1") > 0.9)
    )

    # Condition 6: PS1 source 1, z-band (stricter distance)
    cond6 = (
        (col("distpsnr1") < 10) & (abs_(col("szmag1")) < 14) & (col("sgscore1") > 0.9)
    )

    df = df.withColumn(colname, cond1 | cond2 | cond3 | cond4 | cond5 | cond6)

    return df


def varstar(df, colname="variablesource"):
    """Is likely a variable star"""
    # Dynamic prevpass threshold using a case-like structure
    prevpass_threshold = (
        when(col("age") > 360, lit(1)).when(col("age") > 90, lit(2)).otherwise(lit(3))
    )

    cond1 = (
        (col("age") > 90)
        & (col("ndethist") > 30)
        & (col("prevpasscount") >= prevpass_threshold)
        & (col("magnr") > 0)
        & ~((col("m_now") <= col("peakmag")) & (col("m_now") < 18.5))
        & (
            ((col("distnr") < 0.4) & (col("magnr") < 19.5))
            | ((col("distnr") < 0.8) & (col("magnr") < 17.5))
            | ((col("distnr") < 1.2) & (col("magnr") < 15.5))
        )
    )

    cond2 = (
        (col("maggaia") > 0)
        & (col("neargaia") > 0)
        & (
            ((col("neargaia") < 0.35) & (col("maggaia") < 17) & (col("age") > 30))
            | (
                (col("neargaia") < 0.35)
                & (col("maggaia") < 19)
                & (col("age") > 300)
                & (col("m_now") > 18.5)
            )
            | ((col("neargaia") < 0.2) & (col("maggaia") < 18) & (col("age") > 90))
        )
    )

    cond3 = (
        (col("sgscore1") > 0.25)
        & (col("distpsnr1") < 3)
        & (col("age") > 90)
        & (col("ps1maxmag") < 16)
    )

    cond4 = (
        (col("sgscore1") == 0.5)
        & (col("distpsnr1") < 0.5)
        & (col("age") > 90)
        & (col("ps1maxmag") < 17)
    )

    cond5 = (
        (col("magnr") > 0)
        & (col("magnr") < (col("m_now") - 1))
        & (col("age") > 90)
        & (col("distnr") < 0.5)
        & (col("m_now") > col("peakmag"))
    )

    cond6 = col("ndethist") > 200

    df = df.withColumn(colname, cond1 | cond2 | cond3 | cond4 | cond5 | cond6)

    return df


def is_stationary(df, colname="stationary"):
    """At least one previous detection at this position."""
    df = df.withColumn(
        colname,
        coalesce(
            expr(
                """
              EXISTS(prv_candidates,
                cand ->
                  abs(t_now - cand.jd) > 0.02 AND
                  cand.magpsf < 99 AND
                  (
                    lower(cast(cand.isdiffpos as string)) = '1' OR
                    lower(cast(cand.isdiffpos as string)) = 'true' OR
                    lower(cast(cand.isdiffpos as string)) = 't'
                  )
              )
            """
            ),
            lit(False),
        ),
    )
    return df


@profile
def extract_transient_features(df):
    """Add various columns to the Spark DataFrame

    Notes
    -----
    This information is taken from
    1. https://ui.adsabs.harvard.edu/abs/2020ApJ...904...35P/abstract
    2. https://ui.adsabs.harvard.edu/abs/2020ApJ...895...32F/abstract

    and the original implementation can be found in Zenodo at
    https://zenodo.org/records/4054129

    Additional columns:
    1. faint: magpsf is currently fainter than 19.8, or the source had a very recent detection fainter than 19.
    2. pointunderneath: is likely sitting on top of or blended with a star in Pan-STARRS.
    3. positivesubtraction: is brighter than the template image.
    4. real: is likely a genuine astrophysical transient and not an artifact.
    5. stationary: at least one previous detection at this position.
    6. brightstar: is likely contaminated by a nearby bright star.
    7. variablesource: is likely a variable star

    Parameters
    ----------
    df: Spark DataFrame
        DataFrame with alerts

    Returns
    -------
    out: Spark DataFrame
        Input DataFrame with 8 new columns.

    Examples
    --------
    >>> df = spark.read.load(ztf_alert_sample)
    >>> cols = df.columns

    >>> df2 = extract_transient_features(df)
    >>> cols2 = df2.columns
    >>> assert len(cols2) == len(cols) + 7, (len(cols), len(cols2), cols2)
    """
    # Get initial columns
    cols = df.columns

    # Extract intermediate columns
    df = extract_intermediate_cols(df)

    # Define extra columns
    extra_columns = [
        "faint",
        "positivesubtraction",
        "real",
        "pointunderneath",
        "brightstar",
        "variablesource",
        "stationary",
    ]

    # Add columns
    df = is_faint(df, colname=extra_columns[0])
    df = positive_subtraction(df, colname=extra_columns[1])
    df = real_transient(df, colname=extra_columns[2])
    df = point_underneath(df, colname=extra_columns[3])
    df = bright_star(df, colname=extra_columns[4])
    df = varstar(df, colname=extra_columns[5])
    df = is_stationary(df, colname=extra_columns[6])

    return df.select(cols + extra_columns)


if __name__ == "__main__":
    """Execute the test suite"""

    globs = globals()

    path = os.path.dirname(__file__)
    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
