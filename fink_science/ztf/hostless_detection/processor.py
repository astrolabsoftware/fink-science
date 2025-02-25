# Copyright 2024-2025 AstroLab Software
# Author: R. Durgesh
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
"""Implementation of the paper: ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical Transients https://arxiv.org/abs/2404.18165"""

from line_profiler import profile
import os

import numpy as np
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd

from fink_science.ztf.hostless_detection.run_pipeline import HostLessExtragalactic
from fink_science.ztf.hostless_detection.pipeline_utils import load_json
from fink_science.tester import spark_unit_tests


current_directory = os.path.dirname(os.path.abspath(__file__))
CONFIGS = load_json("{}/config.json".format(current_directory))


@pandas_udf(ArrayType(FloatType()))
@profile
def run_potential_hostless(
    magpsf: pd.Series,
    cutoutScience: pd.Series,
    cutoutTemplate: pd.Series,
    snn_snia_vs_nonia: pd.Series,
    snn_sn_vs_all: pd.Series,
    rf_snia_vs_nonia: pd.Series,
    rf_kn_vs_nonkn: pd.Series,
    finkclass: pd.Series,
    tnsclass: pd.Series,
    deltat: pd.Series,
    roid: pd.Series,
) -> pd.Series:
    """
    Runs potential hostless candidate detection using

    Parameters
    ----------
    magpsf: pd.Series
        Magnitude from PSF-fit photometry [mag]
    cutoutScience: pd.Series
        science stamp images
    cutoutTemplate: pd.Series
        template stamp images
    snn_snia_vs_nonia: pd.Series
        The probability of an alert to be a SN Ia vs.
        core-collapse SNe using SuperNNova
    snn_sn_vs_all: pd.Series
        The probability of an alert to be a SNe vs. anything else
        (variable stars and other categories in the training) using SuperNNova
    rf_snia_vs_nonia: pd.Series
        Probability of an alert to be a SNe Ia using a Random Forest Classifier
        (binary classification). Higher is better
    rf_kn_vs_nonkn: pd.Series
        robability of an alert to be a Kilonova using a PCA & Random Forest
        Classifier (binary classification). Higher is better.
    finkclass: pd.Series
        Fink derived classification tags
    tnsclass: pd.Series
        Tag from cross-referencing with the TNS database
    deltat: pd.Series
        Delta time between `candidate.jd` and the first variation time
         at 3 sigma (`candidate.jdstarthist`).
    roid: pd.Series
        Series containing SSO label (int).
        Each row contains one label.

    Returns
    -------
    pd.Series
        Score for being hostless (float)

    References
    ----------
    1. ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
     Transients
        https://arxiv.org/abs/2404.18165

    Examples
    --------
    >>> from pyspark.sql.functions import lit
    >>> from fink_filters.classification import extract_fink_classification

    >>> df = spark.read.load(sample_file)
    >>> df.count()
    72

    # Compute Fink classification
    >>> cols = [
    ...    "cdsxmatch", "roid", "mulens", "snn_snia_vs_nonia",
    ...    "snn_sn_vs_all", "rf_snia_vs_nonia", "candidate.ndethist",
    ...    "candidate.drb", "candidate.classtar", "candidate.jd",
    ...    "candidate.jdstarthist", "rf_kn_vs_nonkn", "tracklet"]
    >>> df = df.withColumn("finkclass", extract_fink_classification(*cols))

    # Fake TNS classification for the test
    >>> df = df.withColumn("tnsclass", lit("Unknown"))

    # Add a new column
    >>> df = df.withColumn('kstest_static',
    ...     run_potential_hostless(
    ...         df["cmagpsf"],
    ...         df["cutoutScience.stampData"],
    ...         df["cutoutTemplate.stampData"],
    ...         df["snn_snia_vs_nonia"],
    ...         df["snn_sn_vs_all"],
    ...         df["rf_snia_vs_nonia"],
    ...         df["rf_kn_vs_nonkn"],
    ...         df["finkclass"],
    ...         df["tnsclass"],
    ...         df["candidate.jd"] - df["candidate.jdstarthist"],
    ...         df["roid"]))
    >>> df.filter(df.kstest_static[0] >= 0).count()
    0
    """
    # load the configuration file
    hostless_science_class = HostLessExtragalactic(CONFIGS)

    # compute length of the ligtcurves
    number_of_alerts = magpsf.apply(lambda x: np.sum(np.array(x) == np.array(x)))

    # Init values
    kstest_results = []
    default_result = -99.0

    # score conditions
    c0 = snn_snia_vs_nonia >= 0.5
    c1 = snn_sn_vs_all >= 0.5
    c2 = rf_snia_vs_nonia >= 0.5
    c3 = rf_kn_vs_nonkn >= 0.5

    for index in range(cutoutScience.shape[0]):
        # xmatch conditions
        c4 = finkclass[index] in CONFIGS["finkclass"]
        c5 = tnsclass[index] in CONFIGS["tnsclass"]
        c6 = abs(deltat[index]) <= CONFIGS["cutout_timeframe"]
        c7 = magpsf[index][-1] <= CONFIGS["cutout_magnitude"]
        c8 = roid[index] != 3
        if (
            (c0[index] or c1[index] or c2[index] or c3[index] or c4 or c5)
            and c6
            and c7
            and c8
        ):
            if number_of_alerts[index] >= CONFIGS["minimum_number_of_alerts"]:
                science_stamp = cutoutScience[index]
                template_stamp = cutoutTemplate[index]
                kstest_science, kstest_template = (
                    hostless_science_class.process_candidate_fink(
                        science_stamp, template_stamp
                    )
                )
                kstest_results.append([kstest_science, kstest_template])
            else:
                kstest_results.append([default_result, default_result])
        else:
            kstest_results.append([default_result, default_result])
    return pd.Series(kstest_results)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)
    sample_file = (
        "./fink_science/data/alerts/hostless_detection/part-0-0-435829.parquet"
    )
    globs["sample_file"] = sample_file
    spark_unit_tests(globs)
