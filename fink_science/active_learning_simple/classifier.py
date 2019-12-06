# Copyright 2019 AstroLab Software
# Author: Julien Peloton
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
from scipy import optimize
import numpy as np
import pickle

from fink_science.active_learning_simple.bazin import fit_scipy

def extract_field(current: list, history: list) -> np.array:
    """ Concatenate current and historical data
    """
    conc = [np.concatenate((j, [i])) for i, j in zip(current, history)]
    return np.array(conc)

def fit_all_bands(
        times: np.array, fluxes: np.array, bands: np.array) -> np.array:
    """ Perform a Bazin fit for all alerts and all bands.

    For a given set of parameters (a, b, ...), and a given
    set of bands (g, r, ...), the final feature vector must be of the form:

    features = [
        [ga, gb, ... ra, rb, ... ], # alert 0
        [ga, gb, ... ra, rb, ... ], # alert 1
        [ga, gb, ... ra, rb, ... ], # alert ...
    ]

    Parameters
    ----------
    times: 2D np.array (alerts, times)
        Array of time vectors (float)
    fluxes: 2D np.array (alerts, fluxes)
        Array of flux vectors (float).

    Returns
    ----------
    features: 2D np.array (alerts, features x bands)
        Array of feature vectors (all bands for each alert)

    Examples
    --------
    """
    features = []
    unique_bands = [1, 2, 3, 4]
    # Loop over all alerts
    for talert, falert, balert in zip(times, fluxes, bands):
        isNan = falert != falert
        if np.sum(isNan) > 0:
            falert[isNan] = 0.0
        feature_alert = []
        # For each alert, estimate the parameters for each band
        for band in unique_bands:
            mask = np.where(balert == band)[0]
            tband = talert[mask]
            fband = falert[mask]
            if len(tband) < 5:
                feature_alert.extend(np.zeros(5, dtype=np.float))
            else:
                feature_alert.extend(fit_scipy(tband, fband))
        features.append(np.array(feature_alert))
    return np.array(features)

def apply_classifier(
        times: np.array, mags: np.array,
        bands: np.array) -> (np.array, np.array):
    """ Apply classifier on a batch of alert data.

    Parameters
    ----------
    times: 2D np.array (alert, times)
        Array of time vectors (float)
    fluxes: 2D np.array (alert, fluxes)
        Array of flux vectors (float)
    bands: 2D np.array (alerts, bands)
        Array of filter ID vectors (int)

    Returns
    ----------
    predictions: 1D np.array of labels (str)
    probabilities: 1D np.array of floats (btw 0 and 1)
    """
    # Convert mags in fluxes

    # Compute the test_features: fit_all_bands
    test_features = fit_all_bands(times, mags, bands)

    # Load pre-trained model `clf`
    fn = '/Users/julien/Documents/workspace/myrepos/fink-science/fink_science/active_learning_simple/RandomForestResult.obj'
    clf = load_external_model(fn)

    # Make predictions
    predictions = clf.predict(test_features)
    probabilities = clf.predict_proba(test_features)

    return predictions, probabilities


def load_external_model(fn: str = ''):
    """ Load a RandomForestClassifier model from disk (pickled).

    Parameters
    ----------
    fn: str
        Filename. This file should be known from all machines!

    Return
    ----------
    clf: sklearn.ensemble.forest.RandomForestClassifier
    """
    return pickle.load(open(fn, 'rb'))
