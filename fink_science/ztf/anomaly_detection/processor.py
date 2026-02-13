# Copyright 2020-2024 AstroLab Software
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
import numpy.ma as ma

import onnxruntime as rt

from fink_science import __file__
from fink_science.tester import spark_unit_tests
from pyspark.sql.functions import pandas_udf

logger = logging.getLogger(__name__)

MODEL_COLUMNS = [
    "amplitude",
    "anderson_darling_normal",
    "beyond_1_std",
    "chi2",
    "cusum",
    "kurtosis",
    "linear_fit_slope",
    "linear_fit_slope_sigma",
    "linear_trend_noise",
    "linear_trend_sigma",
    "magnitude_percentage_ratio_20_10",
    "magnitude_percentage_ratio_40_5",
    "maximum_slope",
    "median",
    "median_absolute_deviation",
    "median_buffer_range_percentage_10",
    "skew",
    "stetson_K",
    "percent_amplitude",
    "linear_fit_reduced_chi2",
    "inter_percentile_range_10",
    "linear_trend",
    "standard_deviation",
    "weighted_mean",
    "mean",
]

ANOMALY_MODELS = [
    "_beta",
    "_anais",
    "_emille",
    "_julien",
    "_maria",
    "_emille_30days",
    "_varvara",
]  # noqa
NOTIFICATIONS_COUNT = {"_emille": 30, "_emille_30days": 30, "_varvara": 20}
CURVE_LAST_DAYS = {"_emille_30days": 30}


class TwoBandModel:
    """Two band model for anomaly detection"""

    def __init__(self, forest_g, forest_r) -> None:
        self.forest_r = forest_r
        self.forest_g = forest_g

    def anomaly_score(self, data_r, data_g, mask_r, mask_g):
        """
        Calculates anomaly score based on data from two filters (bands).

        The logic is as follows:
        - If data is valid for both bands, return the minimum of the two scores.
        - If data is valid for only one band, return the score for that band.
        - If data is invalid for both bands, return 0.0.

        Parameters
        ----------
        data_r: pd.DataFrame
            Features for the r-band.
        data_g: pd.DataFrame
            Features for the g-band.
        mask_r: pd.Series (bool)
            Mask indicating if the original r-band data was invalid (True) or not (False).
        mask_g: pd.Series (bool)
            Mask indicating if the original g-band data was invalid (True) or not (False).

        Returns
        -------
        np.ndarray
            A 1D array of anomaly scores.
        """
        scores_g_raw = self.forest_g.run(
            None, {"X": data_g.to_numpy().astype(np.float32)}
        )
        scores_r_raw = self.forest_r.run(
            None, {"X": data_r.to_numpy().astype(np.float32)}
        )
        scores_g = np.transpose(scores_g_raw[-1])[0]
        scores_r = np.transpose(scores_r_raw[-1])[0]
        masked_scores_g = ma.array(scores_g, mask=mask_g.to_numpy())
        masked_scores_r = ma.array(scores_r, mask=mask_r.to_numpy())
        final_scores = (
            ma
            .column_stack([masked_scores_g, masked_scores_r])
            .min(axis=1)
            .filled(np.nan)
        )

        return final_scores


@pandas_udf(DoubleType())
def anomaly_score(lc_features, model=None):
    """Returns anomaly score for an observation

    Parameters
    ----------
    lc_features: Spark Map
        Dict of dicts of floats. Keys of first dict - filters (fid), keys of inner dicts - names of features.
    model: str
        Name of the model used.
        Name must start with a `_` and be `_{user_name}`,
        where user_name is the user name of the model at https://anomaly.fink-portal.org/.

    Returns
    -------
    out: float
        Anomaly score

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> from pyspark.sql.functions import isnan, col
    >>> from fink_science.ztf.ad_features.processor import extract_features_ad

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid', 'distnr', 'magnr', 'sigmagnr', 'isdiffpos']

    >>> MODELS = ['', '_beta'] # '' corresponds to the model for a telegram channel
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> cols = ['cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId', 'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']
    >>> df = df.withColumn('lc_features', extract_features_ad(*cols))
    >>> for model in MODELS:
    ...     df = df.withColumn(f'anomaly_score{model}', anomaly_score("lc_features", F.lit(model)))

    >>> df.filter(df["anomaly_score"] < -0.013).count()
    229

    >>> df.filter(isnan(col("anomaly_score"))).count() < 200
    True

    # Check the robustness of the code when i-band is present
    >>> df = spark.read.load(ztf_alert_with_i_band)

    # Required alert columns, concatenated with historical data
    >>> what = ['magpsf', 'jd', 'sigmapsf', 'fid', 'distnr', 'magnr', 'sigmagnr', 'isdiffpos']
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> cols = ['cmagpsf', 'cjd', 'csigmapsf', 'cfid', 'objectId', 'cdistnr', 'cmagnr', 'csigmagnr', 'cisdiffpos']
    >>> df = df.withColumn('lc_features', extract_features_ad(*cols))
    >>> df = df.withColumn("anomaly_score", anomaly_score("lc_features"))

    >>> df.filter(df["anomaly_score"] < 0).count()
    118
    """

    def get_key(x: dict, band: int):
        if x is None or not isinstance(x, dict):
            return pd.Series({k: np.nan for k in MODEL_COLUMNS}, dtype=np.float64)
        if band in x and x[band] is not None and len(x[band]) > 0:
            return pd.Series(x[band])
        else:
            return pd.Series({k: np.nan for k in MODEL_COLUMNS}, dtype=np.float64)

    path = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{path}/data/models/anomaly_detection"

    data_r = lc_features.apply(lambda x: get_key(x, 1))[MODEL_COLUMNS]
    data_g = lc_features.apply(lambda x: get_key(x, 2))[MODEL_COLUMNS]

    mask_r = data_r.isna().any(1)
    mask_g = data_g.isna().any(1)

    if model is not None:
        model = model.to_numpy()[0]
    else:
        model = ""

    g_model_path_AAD = f"{model_path}/forest_g_AAD{model}.onnx"
    r_model_path_AAD = f"{model_path}/forest_r_AAD{model}.onnx"
    if not (os.path.exists(r_model_path_AAD) and os.path.exists(g_model_path_AAD)):
        # unzip in a tmp place
        tmp_path = "/tmp"
        g_model_path_AAD = f"{tmp_path}/forest_g_AAD{model}.onnx"
        r_model_path_AAD = f"{tmp_path}/forest_r_AAD{model}.onnx"
        # check it does not exist to avoid concurrent write
        if not (os.path.exists(g_model_path_AAD) and os.path.exists(r_model_path_AAD)):
            with zipfile.ZipFile(
                f"{model_path}/anomaly_detection_forest_AAD{model}.zip", "r"
            ) as zip_ref:
                zip_ref.extractall(tmp_path)

    forest_r_AAD = rt.InferenceSession(r_model_path_AAD)
    forest_g_AAD = rt.InferenceSession(g_model_path_AAD)

    model_AAD = TwoBandModel(forest_g_AAD, forest_r_AAD)
    score_ = model_AAD.anomaly_score(data_r, data_g, mask_r, mask_g)
    return pd.Series(score_)


if __name__ == "__main__":
    """ Execute the test suite """
    globs = globals()

    path = os.path.dirname(os.path.abspath(__file__))
    ztf_alert_sample = "file://{}/data/alerts/datatest".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    ztf_alert_with_i_band = (
        "file://{}/data/alerts/20240606_iband_history.parquet".format(path)
    )
    globs["ztf_alert_with_i_band"] = ztf_alert_with_i_band

    # Run the test suite
    spark_unit_tests(globs)
