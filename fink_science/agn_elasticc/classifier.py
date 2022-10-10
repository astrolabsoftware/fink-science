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

import pickle
import fink_science.agn_elasticc.kernel as k
import fink_science.agn_elasticc.feature_extraction as fe
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests
import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401


def load_classifier(model_path=None):
    """
    load the random forest classifier trained to recognize the AGN
    on binary cases : AGNs vs non-AGNs  (pickle format).

    Parameters
    ----------
    model_path: str, optional
        Path to the model. If None (default), it is
        taken from `k.CLASSIFIER`.

    Returns
    -------
    RandomForestClassifier

    Examples
    --------
    >>> rf = load_classifier()
    >>> rf.n_classes_
    2
    >>> rf.n_features_
    39
    """
    if model_path is None:
        model_path = k.CLASSIFIER

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    return clf


def agn_classifier(data, model_path=None):
    """
    call the agn_classifier

    Parameters
    ----------
    data : DataFrame
        alerts from fink with aggregated lightcurves
    model_path: str, optional
        Path to the model. If None (default), it is
        taken from `k.CLASSIFIER`.

    Returns
    -------
    np.array
        ordered probabilities of being an AGN
        Return 0 if the minimum number of point per passband is not respected
    """

    formated = fe.format_data(data)

    all_transformed, valid = fe.transform_data(formated, k.MINIMUM_POINTS)

    # Filter empty alerts after filtering
    all_empty = [i.empty for i in all_transformed]
    if np.all(all_empty):
        return np.zeros(len(data), dtype=np.float)

    all_features = fe.parametrise(all_transformed)

    features = fe.merge_features(all_features, k.MINIMUM_POINTS)

    clf = load_classifier(model_path)

    proba = fe.get_probabilities(clf, features, valid)

    return proba


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "{}/data/alerts/agn_elasticc_alerts.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
