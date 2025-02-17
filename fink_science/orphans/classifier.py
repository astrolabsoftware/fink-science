# Copyright 2019-2022 AstroLab Software
# Authors: Marina Masson
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

import os
import pandas as pd
import numpy as np
import joblib

from sklearn import preprocessing

from fink_science.orphans.features_extraction import compute_duration_between_first_and_peak, compute_rates, compute_colours
from fink_science.orphans.basic_functions import clean_and_sort_light_curve
from fink_science.orphans.fit import fit_light_curve


def get_features(ctimemjd, cabmags, cabmagserr, cfilts, valid):
    """
    Computes the features from the detected light curve

    Parameters
    ----------
    ctimemjd: pandas.Series of list of float
        Concatenated time in MJD for the object
    cabmags: pandas.Series of list of float, or Nan
        Concatenated magnitude for the object
    cabmagserr: pandas.Series of list of float, or Nan
        Concatenated errors on the magnitude for the object
    cfilts: pandas.Series of list of int
        Concatenated filters for the object

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame containing the features
    """

    # keep only non-NaN values and sort points by time
    times, mags, err, filts = clean_and_sort_light_curve(ctimemjd, cabmags, cabmagserr, cfilts)

    # duration between the first detection and the peak
    duration = np.array(
        [
            compute_duration_between_first_and_peak(t, m) for t, m, v in zip(times.values, mags.values, valid.values) if v
        ]

    )

    # increase rate and decrease rates (average, in the 1/3 and the 3/3 parts of the light curve)
    rates = np.array(
        [
            compute_rates(t, m, f) for t, m, f, v in zip(times.values, mags.values, filts.values, valid.values) if v
        ]
    )

    # colours for the pairs (g, r) and (r, i)
    colours = np.array(
        [
            compute_colours(t, m, f) for t, m, f, v in zip(times.values, mags.values, filts.values, valid.values) if v
        ]
    )

    # parameters of the fit A, B, C, D and chi2
    fit_parameters = np.array(
        [
            fit_light_curve(t, m, e, f) for t, m, e, f, v in zip(times.values, mags.values, err.values, filts.values, valid.values) if v
        ]
    )

    # gather all the features in one DataFrame
    features = {
        'duration': duration,
        'increase_rate': rates[:, 0],
        'decrease_rate_1': rates[:, 1],
        'decrease_rate_3': rates[:, 2],
        'g-r': colours[:,0],
        'r-i': colours[:,1],
        'A': fit_parameters[:, 0],
        'B': fit_parameters[:, 1],
        'C': fit_parameters[:, 2],
        'D': fit_parameters[:, 3],
        'A/B': fit_parameters[:, 0] / fit_parameters[:, 1],
        'chi2': fit_parameters[:, 4]
    }

    df_features = pd.DataFrame(data=features)

    return df_features


def get_probabilities(df_features, valid):
    """Returns probability of being an orphan afterglow predicted by the classifier

    Parameters
    ----------
    df_features: pd.DataFrame
        Features extracted from the light curves
    valid: np.array
        Bool array, indicates if the light curve contains at least 5 data points (all filters)

    Returns
    -------
    final_proba : np.array
        Probabilities of being an orphan afterglow
        Proba = 0. if the object is not valid
    """

    final_proba = np.array([0.0] * len(valid)).astype(np.float64)

    # load classifier
    curdir = os.path.dirname(os.path.abspath(__file__))
    model_path = curdir + '/../data/models/'
    clf = joblib.load(model_path + 'ml_model_orphans.pkl')

    if len(df_features['duration']) > 0:
        # clean non-valid data
        df_features.replace([np.inf, -np.inf], 1000, inplace=True)
        df_features.fillna(0, inplace=True)

        # normalise the features
        features_norm = preprocessing.normalize(df_features, norm='max')
        proba = clf.predict_proba(features_norm)
        index_to_replace = np.where(valid.values)
        final_proba[index_to_replace] = proba[:, 1]

    return final_proba


def orphan_classifier(ctimemjd, cabmags, cabmagserr, cfilts, valid):
    """
    Call the orphan_classifier

    Parameters
    ----------
    ctimemjd: pandas.Series of list of float
        Concatenated time in MJD for the object
    cabmags: pandas.Series of list of float, or Nan
        Concatenated magnitude for the object
    cabmagserr: pandas.Series of list of float, or Nan
        Concatenated errors on the magnitude for the object
    cfilts: pandas.Series of list of int
        Concatenated filters for the object
    valid: np.array
        Bool array, indicates if the light curve contains at least 5 data points (all filters)

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the proba of an event to be an orphan
    """

    features = get_features(ctimemjd, cabmags, cabmagserr, cfilts, valid)
    proba = get_probabilities(features, valid)

    return proba