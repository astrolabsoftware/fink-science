"""
    Implementation of the paper:
    ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
    Transients
    https://arxiv.org/abs/2404.18165
"""

import os

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType
import pandas as pd

from fink_science.hostless_detection.run_pipeline import HostLessExtragalactic
from fink_science.hostless_detection.pipeline_utils import load_json
from fink_science.tester import spark_unit_tests


CONFIGS = load_json("./fink_science/hostless_detection/config.json")


@pandas_udf(FloatType())
def run_potential_hostless(
        cutoutScience: pd.Series, cutoutTemplate: pd.Series) -> pd.Series:
    """
    Runs potential hostless candidate detection using

    Parameters
    ----------
    cutoutScience
        science stamp images
    cutoutTemplate
        template stamp images

    Returns
    ----------
    results

    References
    ----------
    1. ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
     Transients
        https://arxiv.org/abs/2404.18165

    Examples
    ----------
    >>> columns_to_select = ["cutoutScience", "cutoutTemplate"]
    >>> df = spark.read.load(sample_file)
    >>> df.count()
    72
    >>> df = df.select("cutoutScience", "cutoutTemplate")
    >>> df = df.withColumn('kstest_static', run_potential_hostless(df["cutoutScience"], df["cutoutTemplate"]))
    >>> df = df.select("kstest_static").toPandas()
    >>> len(df[df["kstest_static"] >= 0])
    3
    """
    hostless_science_class = HostLessExtragalactic(CONFIGS)
    results = []
    for index in range(cutoutScience.shape[0]):
        science_stamp = cutoutScience["stampData"][index]
        template_stamp = cutoutTemplate["stampData"][index]
        current_result = hostless_science_class.process_candidate_fink(
            science_stamp, template_stamp)
        results.append(current_result)
    return pd.Series(results)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)
    sample_file = 'file://{}/data/alerts/hostless_detection/part-0-0-435829.parquet'.format(path)
    globs["sample_file"] = sample_file
    spark_unit_tests(globs)
