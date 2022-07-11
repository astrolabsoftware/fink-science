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
        Return -1 if the minimum number of point per passband is not respected

    Examples
    --------
    >>> df = pd.read_parquet(ztf_alert_sample)
    >>> proba = agn_classifier(df)
    >>> len(proba)
    2
    >>> len(proba[proba!=-1])
    1
    >>> len(proba[proba==-1])
    1
    """

    clean = fe.clean_data(data)
    converted = fe.convert_full_dataset(clean)
    transformed_1, transformed_2 = fe.transform_data(converted)

    features_1 = fe.parametrise(transformed_1, k.MINIMUM_POINTS, 1)
    features_2 = fe.parametrise(transformed_2, k.MINIMUM_POINTS, 2)

    features, valid = fe.merge_features(features_1, features_2)

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
