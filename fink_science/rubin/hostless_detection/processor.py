# Copyright 2020-2025 AstroLab Software
# Author: Rupesh Durgesh
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
import os

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType

from fink_science.rubin.hostless_detection.run_pipeline import (
    HostLessExtragalacticRubin,
)
from fink_science.ztf.hostless_detection.pipeline_utils import load_json
from fink_science.tester import spark_unit_tests

CONFIGS_BASE = load_json("fink_science/ztf/hostless_detection/config_base.json")
CONFIGS = load_json("fink_science/ztf/hostless_detection/config.json")
CONFIGS.update(CONFIGS_BASE)


@F.pandas_udf(ArrayType(FloatType()))
@profile
def run_potential_hostless(
    cutoutScience: pd.Series, cutoutTemplate: pd.Series
) -> pd.Series:
    """Runs potential hostless candidate detection for Rubin without any filtering

    Parameters
    ----------
    cutoutScience: pd.Series
        science stamp images
    cutoutTemplate: pd.Series
        template stamp images

    Returns
    -------
    pd.Series
        Scores (array of 2 floats) for being hostless

    References
    ----------
    1. ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
     Transients
        https://arxiv.org/abs/2404.18165

    Examples
    --------
    >>> df = spark.read.format('parquet').load(rubin_alert_sample)
    >>> df.count()
    50

    # Add a new column
    >>> df = df.withColumn('kstest_static',
    ...     run_potential_hostless(
    ...         df["cutoutScience"],
    ...         df["cutoutTemplate"]))
    >>> df.filter(df.kstest_static[0] >= 0).count()
    3
    """
    kstest_results = []
    hostless_science_class = HostLessExtragalacticRubin(CONFIGS_BASE)

    for index in range(cutoutScience.shape[0]):
        science_stamp = cutoutScience[index]
        template_stamp = cutoutTemplate[index]
        kstest_science, kstest_template = (
            hostless_science_class.process_candidate_fink_rubin(
                science_stamp, template_stamp
            )
        )
        kstest_results.append([kstest_science, kstest_template])
    return pd.Series(kstest_results)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    rubin_alert_sample = "./fink_science/data/alerts/or4_lsst7.1"
    globs["rubin_alert_sample"] = rubin_alert_sample
    spark_unit_tests(globs)
