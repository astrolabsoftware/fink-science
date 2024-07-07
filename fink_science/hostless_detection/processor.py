"""
    Implementation of the paper:
    ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical
    Transients
    https://arxiv.org/abs/2404.18165
"""

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType
import pandas as pd

from run_pipeline import HostLessExtragalactic
from science.pipeline_utils import load_json

CONFIGS = load_json("config.json")


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
    """
    hostless_science_class = HostLessExtragalactic(CONFIGS)
    results = []
    for index in range(cutoutScience.shape[0]):
        science_stamp = cutoutScience[index]
        template_stamp = cutoutTemplate[index]
        current_result = hostless_science_class.process_candidate_fink(
            science_stamp, template_stamp)
        results.append(current_result)
    return pd.Series(results)
