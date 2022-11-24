# Copyright 2022 Fink Software
# Author: Etienne Russeil
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

import joblib
import fink_science.pisn.kernel as k
import fink_science.agn.feature_extraction as fe_agn
import fink_science.pisn.feature_extraction as fe_pisn
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests
import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401


def load_classifier():
    """
    load the random forest classifier trained to recognize the PISN
    on binary cases : PISN vs non-PISN  (joblib format).

    Returns
    -------
    RandomForestClassifier

    Examples
    --------
    """

    model_path = k.CLASSIFIER_ELASTICC
    clf = joblib.load(k.CLASSIFIER_ELASTICC)

    return clf


def pisn_classifier(data):
    """
    Call the pisn_classifier

    Parameters
    ----------
    data : DataFrame
        alerts from fink with aggregated lightcurves

    Returns
    -------
    np.array
        ordered probabilities of being an PISN
        Return 0 if the minimum number of point per passband is not respected

    Examples
    --------
    """

    formated = fe_agn.format_data(data, 'ELASTICC')

    all_transformed, valid = fe_pisn.transform_data(formated)

    if not valid.any():
        return np.zeros(len(data), dtype=np.float)

    all_features = fe_pisn.parametrise(all_transformed)

    features = fe_pisn.merge_features(all_features)

    clf = load_classifier()

    proba = fe_agn.get_probabilities(clf, features, valid)

    return proba


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "{}/data/alerts/agn_elasticc_alerts.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
