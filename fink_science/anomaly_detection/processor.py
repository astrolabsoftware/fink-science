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

from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

import pandas as pd
import numpy as np

from onnx import load
import onnxruntime as rt


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
        scores_g = self.forest_g.run(None, {"X": data_g.values.astype(np.float32)})
        scores_r = self.forest_r.run(None, {"X": data_r.values.astype(np.float32)})
        return (scores_g[-1] + scores_r[-1]) / 2


path = os.path.dirname(os.path.abspath(__file__))
model_path = f"{path}/data/models/anomaly_detection"
g_model_path = f"{model_path}/forest_g.onnx"
r_model_path = f"{model_path}/forest_r.onnx"
g_model_path_AAD = f"{model_path}/forest_g_AAD.onnx"
r_model_path_AAD = f"{model_path}/forest_r_AAD.onnx"
if not (os.path.exists(r_model_path) and os.path.exists(g_model_path)):
    # unzip in a tmp place
    tmp_path = '/tmp'
    g_model_path = f"{tmp_path}/forest_g.onnx"
    r_model_path = f"{tmp_path}/forest_r.onnx"
    g_model_path_AAD = f"{tmp_path}/forest_g_AAD.onnx"
    r_model_path_AAD = f"{tmp_path}/forest_r_AAD.onnx"
    # check it does not exist to avoid concurrent write
    if not (os.path.exists(r_model_path) and os.path.exists(g_model_path)):
        with zipfile.ZipFile(f"{model_path}/anomaly_detection_forest.zip", 'r') as zip_ref:
            zip_ref.extractall(tmp_path)
    if not (os.path.exists(g_model_path_AAD) and os.path.exists(r_model_path_AAD)):
        with zipfile.ZipFile(f"{model_path}/anomaly_detection_forest_AAD.zip", 'r') as zip_ref:
            zip_ref.extractall(tmp_path)


class WrapInferenceSession:
    """
    The class is an additional wrapper over InferenceSession
    to solve the pyspark serialisation problem

    https://github.com/microsoft/onnxruntime/pull/800#issuecomment-844326099
    """
    def __init__(self, onnx_bytes):
        self.sess = rt.InferenceSession(onnx_bytes.SerializeToString())
        self.onnx_bytes = onnx_bytes

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'onnx_bytes': self.onnx_bytes}

    def __setstate__(self, values):
        self.onnx_bytes = values['onnx_bytes']
        self.sess = rt.InferenceSession(self.onnx_bytes.SerializeToString())


forest_r = WrapInferenceSession(load(r_model_path))
forest_g = WrapInferenceSession(load(g_model_path))
forest_r_AAD = WrapInferenceSession(load(r_model_path_AAD))
forest_g_AAD = WrapInferenceSession(load(g_model_path_AAD))


r_means = pd.read_csv(f"{model_path}/r_means.csv", header=None, index_col=0, squeeze=True)
g_means = pd.read_csv(f"{model_path}/g_means.csv", header=None, index_col=0, squeeze=True)

model = TwoBandModel(forest_g, forest_r)
model_AAD = TwoBandModel(forest_g_AAD, forest_r_AAD)


def anomaly_score(lc_features, model_type='AADForest'):
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
        >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid', 'distnr', 'magnr', 'sigmagnr', 'isdiffpos']
        >>> prefix = 'c'
        >>> what_prefix = [prefix + i for i in what]
        >>> for colname in what:
        ...    df = concat_col(df, colname, prefix=prefix)

        >>> cols = ['cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId', 'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']
        >>> df = df.withColumn('lc_features', extract_features_ad(*cols))
        >>> df = df.withColumn("anomaly_score", anomaly_score("lc_features"))

        >>> df.filter(df["anomaly_score"] < -0.5).count()
        7

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
        if model_type == 'AADForest':
            return model_AAD.anomaly_score(data_r, data_g)[0].item()
        return model.anomaly_score(data_r, data_g)[0].item()
    return anomaly_score(lc_features)

@pandas_udf(DoubleType())
def anomaly_score_vect(lc_features, model_type="AADForest"):
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
    >>> df = df.withColumn("anomaly_score", anomaly_score_vect("lc_features"))

    >>> df.filter(df["anomaly_score"] < -0.5).count()
    7

    >>> df.filter(df["anomaly_score"] == 0).count()
    84
    """

    def get_key(x, band):
        if (
            len(x) != 2
            or x is None
            or any(
                map(  # noqa: W503
                    lambda fs: (fs is None or len(fs) == 0), x.values()
                )
            )
        ):
            return pd.Series({}, dtype=np.float64)
        else:
            return pd.Series(x[band])

    data_r = lc_features.apply(lambda x: get_key(x, 1)).fillna(0)[MODEL_COLUMNS]
    data_g = lc_features.apply(lambda x: get_key(x, 2)).fillna(0)[MODEL_COLUMNS]

    if model_type == "AADForest":
        score = model_AAD.anomaly_score(data_r, data_g)
    score = model.anomaly_score(data_r, data_g)
    return pd.Series(np.transpose(score)[0])


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
