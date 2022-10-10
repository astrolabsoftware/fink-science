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
import fink_science.agn.kernel as k
import fink_science.agn.feature_extraction as fe
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests
import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401


def load_classifier():
    """
    load the random forest classifier trained to recognize the AGN
    on binary cases : AGNs vs non-AGNs  (pickle format).

    Returns
    -------
    RandomForestClassifier

    Examples
    --------
    >>> rf = load_classifier()
    >>> rf.n_classes_
    2
    >>> rf.n_features_
    12
    """
    with open(k.CLASSIFIER, "rb") as f:
        clf = pickle.load(f)

    return clf


def agn_classifier(data):
    """
    call the agn_classifier

    Parameters
    ----------
    data : DataFrame
        alerts from fink with aggregated lightcurves

    Returns
    -------
    np.array
        ordered probabilities of being an AGN
        Return 0.0 if the minimum number of point per passband is not respected

    Examples
    --------
    >>> df = pd.read_parquet(ztf_alert_sample)
    >>> proba = agn_classifier(df)
    >>> len(proba)
    2
    >>> len(proba[proba != 0.0])
    1
    >>> len(proba[proba == 0.0])
    1
    """

    clean = fe.clean_data(data)
    converted = fe.convert_full_dataset(clean)

    transformed_1, transformed_2, valid = fe.transform_data(converted, k.MINIMUM_POINTS)

    all_empty = transformed_1.empty | transformed_2.empty
    if all_empty:
        return np.zeros(len(data), dtype=np.float)

    features_1 = fe.parametrise(transformed_1, 1)
    features_2 = fe.parametrise(transformed_2, 2)

    features = fe.merge_features(features_1, features_2)

    clf = load_classifier()

    proba = fe.get_probabilities(clf, features, valid)

    return proba


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "{}/data/alerts/agn_example.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
