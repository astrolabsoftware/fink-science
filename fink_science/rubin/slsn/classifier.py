# Copyright 2022-2024 Fink Software
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
from line_profiler import profile

import joblib
import fink_science.rubin.slsn.kernel as k
import fink_science.rubin.slsn.feature_extraction as fe
from fink_science.tester import regular_unit_tests
import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401


def load_classifier(metadata):
    """Load the random forest classifier (joblib format).

    Returns
    -------
    RandomForestClassifier

    """
    if metadata:
        clf = joblib.load(k.CLASSIFIER_ELASTICC_WITH_MD)

    else:
        clf = joblib.load(k.CLASSIFIER_ELASTICC_NO_MD)

    return clf


@profile
def get_probabilities(clf, features, valid):
    """Returns probabilty of being an SLSN predicted by the classifier.

    Parameters
    ----------
    clf: RandomForestClassifier
        Binary SLSN vs non SLSN classifier.
    features: pd.DataFrame
        Features extracted from the objects.
        Outputed by merge_features.
    valid: np.array
        Bool array, indicates if both passband respect the minimum number of points.

    Returns
    -------
    final_proba : np.array
        ordered probabilities of being a SLSN.
        Proba = 0 if the object is not valid.
    """
    final_proba = np.array([0.0] * len(valid)).astype(np.float64)

    if len(features) > 0:
        features = features.replace(np.inf, 0.0).replace(np.nan, 0.0)
        agn_or_not = clf.predict_proba(features.iloc[:, 1:])
        index_to_replace = features.iloc[:, 1:].index
        final_proba[index_to_replace.to_numpy()] = agn_or_not[:, 1]

    return final_proba


@profile
def slsn_classifier(data, metadata):
    """
    Call the slsn_classifier

    Parameters
    ----------
    data : DataFrame
        alerts from fink with aggregated lightcurves

    Returns
    -------
    np.array
        ordered probabilities of being an slsn
        Return 0 if the minimum number of point per passband is not respected

    """
    transformed, valid = fe.transform_data(data)

    if not valid.any():
        return np.zeros(len(data), dtype=float)

    features = fe.parametrise(transformed[valid], metadata)
    clf = load_classifier(metadata)
    proba = get_probabilities(clf, features, valid)

    return proba


if __name__ == "__main__":
    globs = globals()

    # Run the test suite
    regular_unit_tests(globs)
