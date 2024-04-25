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
import zipfile

from pyspark.sql.types import DoubleType

import pandas as pd
import numpy as np

import onnxruntime as rt

from fink_science import __file__
from fink_science.tester import spark_unit_tests
from pyspark.sql.functions import pandas_udf

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

    def anomaly_score(self, data_r, data_g):
        scores_g = self.forest_g.run(None, {"X": data_g.values.astype(np.float32)})
        scores_r = self.forest_r.run(None, {"X": data_r.values.astype(np.float32)})
        return (scores_g[-1] + scores_r[-1]) / 2


@pandas_udf(DoubleType())
def anomaly_score(lc_features, model=None):
    """Returns anomaly score for an observation

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
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid', 'distnr', 'magnr', 'sigmagnr', 'isdiffpos']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> cols = ['cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId', 'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']
    >>> df = df.withColumn('lc_features', extract_features_ad(*cols))
    >>> df = df.withColumn(f"anomaly_score", anomaly_score("lc_features"))

    >>> df.filter(df["anomaly_score"] < -0.013).count()
    108

    >>> df.filter(df["anomaly_score"] == 0).count()
    84
    """

    def get_key(x, band):
        if (
                len(x) != 2 or x is None or any(
            map(  # noqa: W503
                lambda fs: (fs is None or len(fs) == 0), x.values()
            )
        )
        ):
            return pd.Series({k: np.nan for k in MODEL_COLUMNS}, dtype=np.float64)
        else:
            return pd.Series(x[band])

    path = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{path}/data/models/anomaly_detection"

    r_means = pd.read_csv(f"{model_path}/r_means.csv", header=None, index_col=0, squeeze=True)
    g_means = pd.read_csv(f"{model_path}/g_means.csv", header=None, index_col=0, squeeze=True)
    data_r = lc_features.apply(lambda x: get_key(x, 1))[MODEL_COLUMNS]
    data_g = lc_features.apply(lambda x: get_key(x, 2))[MODEL_COLUMNS]

    mask_r = data_r.isnull().all(1)
    mask_g = data_g.isnull().all(1)
    mask = mask_r.values * mask_g.values
    if model is not None:
        model = model[0]
    else:
        model = ''

    for col in data_r.columns[data_r.isna().any()]:
        data_r[col].fillna(r_means[col], inplace=True)

    for col in data_g.columns[data_g.isna().any()]:
        data_g[col].fillna(g_means[col], inplace=True)

    g_model_path_AAD = f"{model_path}/forest_g_AAD{model}.onnx"
    r_model_path_AAD = f"{model_path}/forest_r_AAD{model}.onnx"
    if not (os.path.exists(r_model_path_AAD) and os.path.exists(g_model_path_AAD)):
        # unzip in a tmp place
        tmp_path = '/tmp'
        g_model_path_AAD = f"{tmp_path}/forest_g_AAD{model}.onnx"
        r_model_path_AAD = f"{tmp_path}/forest_r_AAD{model}.onnx"
        # check it does not exist to avoid concurrent write
        if not (os.path.exists(g_model_path_AAD) and os.path.exists(r_model_path_AAD)):
            with zipfile.ZipFile(f"{model_path}/anomaly_detection_forest_AAD{model}.zip", 'r') as zip_ref:
                zip_ref.extractall(tmp_path)

    forest_r_AAD = rt.InferenceSession(r_model_path_AAD)
    forest_g_AAD = rt.InferenceSession(g_model_path_AAD)

    # load the mean values used to replace Nan values from the features extraction
    r_means = pd.read_csv(f"{model_path}/r_means.csv", header=None, index_col=0, squeeze=True)
    g_means = pd.read_csv(f"{model_path}/g_means.csv", header=None, index_col=0, squeeze=True)

    model_AAD = TwoBandModel(forest_g_AAD, forest_r_AAD)

    score = model_AAD.anomaly_score(data_r, data_g)
    score_ = np.transpose(score)[0]
    score_[mask] = 0.0
    return pd.Series(score_)


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()

    path = os.path.dirname(os.path.abspath(__file__))
    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
