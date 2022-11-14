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
import fink_science.agn.unit_examples as uex  # noqa: F401


def load_classifier(source):
    """
    load the random forest classifier trained to recognize the AGN
    on binary cases : AGNs vs non-AGNs  (pickle format).

    Parameters
    ----------
    source: string
        Origin of the data.
        Currently accepts 'ZTF' or 'ELASTICC'.

    Returns
    -------
    RandomForestClassifier

    Examples
    --------
    >>> rf_ELASTICC = load_classifier('ELASTICC')
    >>> rf_ELASTICC.n_classes_
    2
    >>> rf_ELASTICC.n_features_
    31
    >>> rf_ZTF = load_classifier('ZTF')
    >>> rf_ZTF.n_classes_
    2
    >>> rf_ZTF.n_features_
    12
    """

    if source == 'ELASTICC':
        model_path = k.CLASSIFIER_ELASTICC
    elif source == 'ZTF':
        model_path = k.CLASSIFIER_ZTF

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    return clf


def agn_classifier(data, source):
    """
    Call the agn_classifier

    Parameters
    ----------
    data : DataFrame
        alerts from fink with aggregated lightcurves
    source: string
        Origin of the data.
        Currently accepts 'ZTF' or 'ELASTICC'.

    Returns
    -------
    np.array
        ordered probabilities of being an AGN
        Return 0 if the minimum number of point per passband is not respected

    Examples
    --------
    >>> df = uex.raw_ztf_unit
    >>> proba = agn_classifier(df, 'ZTF')
    >>> proba[0] == 0
    True
    >>> proba[1] != 0
    True
    """

    formated = fe.format_data(data, source)

    all_transformed, valid = fe.transform_data(formated, k.MINIMUM_POINTS, source)

    if not valid.any():
        return np.zeros(len(data), dtype=np.float)

    all_features = fe.parametrise(all_transformed, source)

    features = fe.merge_features(all_features, k.MINIMUM_POINTS, source)

    clf = load_classifier(source)

    proba = fe.get_probabilities(clf, features, valid)

    return proba


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "{}/data/alerts/agn_elasticc_alerts.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
