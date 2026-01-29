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
from pyspark.sql.types import FloatType, MapType, StringType

from fink_science.rubin.hostless_detection.run_pipeline import (
    HostLessExtragalacticRubin,
)
from fink_science.ztf.hostless_detection.pipeline_utils import load_json
from fink_science.tester import spark_unit_tests

from fink_science import __file__


# Share configuration with ZTF
CONFIGS_BASE = load_json(
    "{}/ztf/hostless_detection/config_base.json".format(os.path.dirname(__file__))
)
CONFIGS = load_json(
    "{}/ztf/hostless_detection/config.json".format(os.path.dirname(__file__))
)
CONFIGS.update(CONFIGS_BASE)


@F.pandas_udf(MapType(StringType(), FloatType()))
@profile
def run_potential_hostless(
    cutoutScience: pd.Series, cutoutTemplate: pd.Series, ssObjectId: pd.Series
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
    25

    # Add a new column
    >>> df = df.withColumn('elephant_kstest',
    ...     run_potential_hostless(
    ...         df["cutoutScience"],
    ...         df["cutoutTemplate"],
    ...         df["ssSource.ssObjectId"]))
    >>> df.filter(df.elephant_kstest.kstest_science >= 0).count()
    0
    >>> df.filter(df.elephant_kstest.kstest_science <= 0).count()
    25
    """
    default_result = {"kstest_science": -99, "kstest_template": -99}
    kstest_results = []
    hostless_science_class = HostLessExtragalacticRubin(CONFIGS_BASE)
    for index in range(cutoutScience.shape[0]):
        if ssObjectId[index] is None:
            science_stamp = cutoutScience[index]
            template_stamp = cutoutTemplate[index]
            kstest_science, kstest_template = (
                hostless_science_class.process_candidate_fink_rubin(
                    science_stamp, template_stamp
                )
            )
            kstest_results.append({
                "kstest_science": kstest_science,
                "kstest_template": kstest_template,
            })
        else:
            kstest_results.append(default_result)
    return pd.Series(kstest_results)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    rubin_alert_sample = "file://{}/data/alerts/hostless_detection/rubin_sample_data_10_0.parquet".format(
        path
    )
    globs["rubin_alert_sample"] = rubin_alert_sample
    spark_unit_tests(globs)
