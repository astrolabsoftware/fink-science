"""
    Implementation of the paper:
    ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
    Transients
    https://arxiv.org/abs/2404.18165
"""

import os

import numpy as np
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType
import pandas as pd

from fink_science.hostless_detection.run_pipeline import HostLessExtragalactic
from fink_science.hostless_detection.pipeline_utils import load_json
from fink_science.tester import spark_unit_tests


current_directory = os.path.dirname(os.path.abspath(__file__))
CONFIGS = load_json("{}/config.json".format(current_directory))


@pandas_udf(FloatType())
def run_potential_hostless(
        magpsf: pd.Series, cutoutScience: pd.Series,
        cutoutTemplate: pd.Series) -> pd.Series:
    """
    Runs potential hostless candidate detection using

    Parameters
    ----------
    magpsf
        Magnitude from PSF-fit photometry [mag]
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
    >>> columns_to_select = ["cmagpsf", "cutoutScience", "cutoutTemplate"]
    >>> df = spark.read.load(sample_file)
    >>> df.count()
    72
    >>> df = df.select(columns_to_select)
    >>> df = df.withColumn('kstest_static', run_potential_hostless(df["cmagpsf"], df["cutoutScience"], df["cutoutTemplate"]))
    >>> df = df.select(["kstest_static"]).toPandas()
    >>> len(df[df["kstest_static"] >= 0])
    3
    """
    hostless_science_class = HostLessExtragalactic(CONFIGS)
    number_of_alerts = magpsf.apply(
        lambda x: np.sum(np.array(x) == np.array(x)))
    results = []
    default_result = -99
    for index in range(cutoutScience.shape[0]):
        science_stamp = cutoutScience["stampData"][index]
        template_stamp = cutoutTemplate["stampData"][index]
        if number_of_alerts[index] >= CONFIGS["minimum_number_of_alerts"]:
            current_result = hostless_science_class.process_candidate_fink(
                science_stamp, template_stamp)
            results.append(current_result)
        else:
            results.append(default_result)
    return pd.Series(results)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)
    sample_file = './fink_science/data/alerts/hostless_detection/part-0-0-435829.parquet'
    globs["sample_file"] = sample_file
    spark_unit_tests(globs)
