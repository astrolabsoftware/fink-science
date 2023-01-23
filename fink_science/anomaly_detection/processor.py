# Copyright 2020-2022 AstroLab Software
# Author: Igor Beschastnov
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
import logging
import os
import pickle
import zipfile

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

import pandas as pd

from fink_science import __file__
from fink_science.tester import spark_unit_tests

logger = logging.getLogger(__name__)

MODEL_COLUMNS = [
    'amplitude', 'anderson_darling_normal', 'beyond_1_std', 'chi2', 'cusum',
    'kurtosis', 'linear_fit_slope', 'linear_fit_slope_sigma',
    'linear_trend_noise', 'linear_trend_sigma',
    'magnitude_percentage_ratio_20_10', 'magnitude_percentage_ratio_40_5',
    'maximum_slope', 'median', 'median_absolute_deviation',
    'median_buffer_range_percentage_10', 'skew', 'stetson_K',
]


class TwoBandModel:
    def __init__(self, forest_g, forest_r) -> None:
        self.forest_r = forest_r
        self.forest_g = forest_g

    def anomaly_score(self, data_g, data_r):
        scores_g = self.forest_g.score_samples(data_g)
        scores_r = self.forest_r.score_samples(data_r)
        return (scores_g + scores_r) / 2


path = os.path.dirname(os.path.abspath(__file__))
model_path = f"{path}/data/models/anomaly_detection"
g_model_path = f"{model_path}/forest_g.pickle"
r_model_path = f"{model_path}/forest_r.pickle"
if not (os.path.exists(r_model_path) and os.path.exists(g_model_path)):
    with zipfile.ZipFile(f"{model_path}/anomaly_detection_forest.zip", 'r') as zip_ref:
        zip_ref.extractall(model_path)

with open(r_model_path, 'rb') as forest_file:
    forest_r = pickle.load(forest_file)
with open(g_model_path, 'rb') as forest_file:
    forest_g = pickle.load(forest_file)
r_means = pd.read_csv(f"{model_path}/r_means.csv", header=None, index_col=0, squeeze=True)
g_means = pd.read_csv(f"{model_path}/g_means.csv", header=None, index_col=0, squeeze=True)

model = TwoBandModel(forest_g, forest_r)


@udf(returnType=DoubleType())
def anomaly_score(lc_features) -> float:
    """ Returns anomaly score for an observation

    Parameters
    ----------
    lc_features: Spark Map
        Dict of dicts of floats. Keys of first dict - filters (fid), keys of inner dicts - names of features.

    Returns
    ----------
    out: float
        Anomaly score

    Examples
    ---------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> from fink_science.ad_features.processor import extract_features_ad

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> df = df.withColumn('lc_features', extract_features_ad(*what_prefix, 'objectId'))
    >>> df = df.withColumn("anomaly_score", anomaly_score("lc_features"))

    >>> df.filter(df["anomaly_score"] < -0.5).count()
    5

    >>> df.filter(df["anomaly_score"] == 0).count()
    84

    """

    if (
        lc_features is None
        or len(lc_features) != 2  # noqa: W503 (https://www.flake8rules.com/rules/W503.html, https://www.flake8rules.com/rules/W504.html)
        or any(map(  # noqa: W503
            lambda fs: (fs is None or len(fs) == 0),
            lc_features.values()
        ))
    ):
        return 0.0
    if any(map(lambda fid: fid not in lc_features, (1, 2))):
        logger.exception(f"Unsupported 'lc_features' format in '{__file__}/{anomaly_score.__name__}'")

    data_r, data_g = (
        pd.DataFrame.from_dict({k: [v] for k, v in lc_features[i].items()})[MODEL_COLUMNS]
        for i in (1, 2)
    )
    for data, means in ((data_r, r_means), (data_g, g_means)):
        for col in data.columns[data.isna().any()]:
            data[col].fillna(means[col], inplace=True)
    return model.anomaly_score(data_r, data_g)[0].item()


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
