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
from fink_science.active_learning_simple.conversion import mag2flux

def extract_field(current: list, history: list) -> np.array:
    """ Concatenate current and historical data.

    If t1 is the first time the object has been seen, and the object has N
    historical measurements, the routine returns values ordered as:
    [t1, t2, ...., tN, current] (past to current).

    Parameters
    ----------
    current: list [nalert, 1]
        List of field values. each entry corresponds to the measurement for
        one alert.
    history: list of list [nalerts, Ndays]
        List of historical field values. Each entry is a list of historical
        measurements for one alert.

    Returns
    ----------
    conc: 2D np.array [nalert, Ndays + 1]
        Array of array. Each entry is an array of historical+current
        measurements for one alert.
    """
    conc = [np.concatenate((j, [i])) for i, j in zip(current, history)]
    return np.array(conc)

def fit_all_bands(
        jd, fid, magpsf, sigmapsf, magnr,
        sigmagnr, magzpsci, isdiffpos) -> np.array:
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
    unique_bands = [1, 2, 3]
    # Loop over all alerts
    zz = zip(jd, fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos)
    for alert_data in zz:
        (ajd, afid, amagpsf, asigmapsf,
            amagnr, asigmagnr, amagzpsci, aisdiffpos) = alert_data

        feature_alert = []
        # For each alert, estimate the parameters for each band
        for band in unique_bands:
            maskband = afid == band
            masknan = amagpsf == amagpsf
            masknone = amagpsf != None
            mask = maskband * masknan * masknone
            if len(ajd[mask]) < 5:
                feature_alert.extend(np.zeros(5, dtype=np.float))
            else:
                # Compute flux
                flux, sigmaflux = mag2flux(
                    band, amagpsf[mask], asigmapsf[mask],
                    amagnr[mask], asigmagnr[mask],
                    amagzpsci[mask], aisdiffpos[mask])
                feature_alert.extend(fit_scipy(ajd[mask], flux))
        features.append(np.array(feature_alert))
    return np.array(features)

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