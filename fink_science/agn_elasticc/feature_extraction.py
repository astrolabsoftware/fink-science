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

import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
import fink_science.agn_elasticc.models as mod
from pandas.testing import assert_frame_equal  # noqa: F401
import fink_science.agn_elasticc.kernel as k  # noqa: F401
import numpy as np
import pickle  # noqa: F401


def map_fid(ps):
    band_dict = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}
    return np.array(list(map(band_dict.get, ps)))


def compute_hostgal_dist(df):

    if (df["hostgal_ra"] == -999) & (df["hostgal_dec"] == -999):
        hostgal_dist = -9
    else:
        hostgal_dist = (
            np.sqrt((df["ra"] - df["hostgal_ra"]) ** 2 + (df["dec"] - df["hostgal_dec"]) ** 2) * 1e3
        )

    return hostgal_dist


def format_data(df):

    # Compute distance from host
    df["hostgal_dist"] = df.apply(compute_hostgal_dist, axis=1)
    df = df.drop(columns={"hostgal_ra", "hostgal_dec"})

    # Transform band str to int
    df["cfid"] = df["cfid"].apply(map_fid)

    return df


def keep_filter(ps, band):
    """
    funtion that removes points from other bands than the one specified

    Parameters
    ---------
    ps : pd.Series
        each rows of the dataframe. each entries must be numeric list

    Return
    ------
    list_with_oneband : list
        list of the same size as x, each entries is the original list from the
        current rows with only the wanted filter and the associated values from the other columns.

    Example
    -------
    >>> example = pd.Series(data = {'cfid':np.array([1, 1, 2, 1, 2]), 'anything':np.array([-2, 86.9, 58.1, 24, 42])})
    >>> filtered = keep_filter(example, 2)
    >>> (np.array_equal(filtered[0], np.array([2, 2]))) & (np.array_equal(filtered[1], np.array([58.1, 42])))
    True
    >>> example2 = pd.Series(data = {'cfid':np.array([2, 2]), 'anything':np.array([24, 42])})
    >>> filtered2 = keep_filter(example2, 1)
    >>> (np.array_equal(filtered2[0], np.array([]))) & (np.array_equal(filtered2[1], np.array([])))
    True

    """

    mask = ps["cfid"] == band

    return [np.array(_col)[mask].astype(type(_col[0])) for _col in ps]


def translate(ps):

    """Translate a cjd list by substracting maxflux point

    Parameters
    ----------
    ps: pd.Series
        Must contain ['cjd', 'cflux']

    Returns
    -------
    np.array
        Translated array. Returns empty array if input was empty

    Example
    -------
    >>> example = pd.Series(data = {'cjd':np.array([1,2,3]), 'cflux':np.array([-2, 42, 23]), 'anything':np.array(['toto', 82, -8])})
    >>> np.array_equal(translate(example), np.array([-1,  0,  1]))
    True
    >>> example2 = pd.Series(data = {'cjd':np.array([]), 'cflux':np.array([]), 'anything':np.array(['toto', 82, -8])})
    >>> np.array_equal(translate(example2), np.array([]))
    True

    """

    if len(ps["cjd"]) == 0:
        return []

    else:
        return ps["cjd"] - ps["cjd"][np.argmax(ps["cflux"])]


def normalize(ps):

    """Normalize by dividing by a data frame of maximum

    Parameters
    ----------
    ps: pd.Series
        Must contain 'cflux', 'csigflux' and 'peak'
    maxi: np.array
        array of all maximum values. -1 if alert has no points

    Returns
    -------
    pd.Series
        Dataframe with columns 'cflux' and 'csigflux' normalized

    Example
    -------
    >>> example = pd.Series(data = {'cflux':np.array([17, 35.7, -3]), 'csigflux':np.array([0.7, 1, 0]), 'peak':35.7})
    >>> out = normalize(example)
    >>> np.array_equal(np.round(out[0], 3), np.array([ 0.476,  1.   , -0.084]))
    True
    >>> np.array_equal(np.round(out[1], 3), np.array([0.02 , 0.028, 0.   ]))
    True
    >>> example2 = pd.Series(data = {'cflux':np.array([]), 'csigflux':np.array([]), 'peak':-1})
    >>> out2 = normalize(example2)
    >>> (np.array_equal(out2[0], np.array([]))) & (np.array_equal(out2[1], np.array([])))
    True
    """

    if len(ps["cflux"]) == 0:
        return ps[["cflux", "csigflux"]]

    else:
        ps["cflux"] = ps["cflux"] / ps["peak"]
        ps["csigflux"] = ps["csigflux"] / ps["peak"]
        return ps[["cflux", "csigflux"]]


def get_max(x, absolute=False):

    """Returns maximum of an array. Returns -1 if array is empty

    Parameters
    ----------
    x: np.array

    absolute: bool
        If true returns absolute maximum
        Default is False

    Returns
    -------
    float
        Maximum of the array or -1 if array is empty

    Example
    -------
    >>> get_max(np.array([1, 78, -6])) == 78
    True
    >>> get_max(np.array([])) == -1
    True
    >>> get_max(np.array([1, 8, -62]), absolute=True) == -62
    True

    """

    if len(x) == 0:
        return -1

    elif absolute:
        return max(x, key=abs)

    else:
        return x.max()


def get_min(x, absolute=False):

    """Returns minimum of an array. Returns -1 if array is empty

    Parameters
    ----------
    x: np.array

    absolute: bool
        If true returns absolute minimum
        Default is False

    Returns
    -------
    float
        Minimum of the array or -1 if array is empty

    Example
    -------
    >>> get_min(np.array([1, 78, -6])) == -6
    True
    >>> get_min(np.array([])) == -1
    True
    >>> get_min(np.array([1, 8, -62]), absolute=True) == 1
    True

    """

    if len(x) == 0:
        return -1

    elif absolute:
        return min(x, key=abs)

    else:
        return x.min()


def transform_data(converted, minimum_points):

    """Apply transformations for each band on a flux converted dataset
            - Shift cjd so that the max flux point is at 0
            - Normalize by dividing flux and flux err by the maximum flux
            - Add a column with maxflux before normalization


    Parameters
    ----------
    converted : pd.DataFrame
        Dataframe of alerts from Fink with nan removed and converted to flux
    minimum_points: minimum number of points for passband to be considered valid
        The classifier requieres at least two consecutive valid passbands

    Returns
    -------
    transformed_1 : pd.DataFrame
        Transformed DataFrame that only contains passband g


    transformed_2 : pd.DataFrame
        Transformed DataFrame that only contains passband r

    """

    all_transformed = []

    for band in range(6):
        transformed = converted.copy()
        transformed[["cfid", "cjd", "cflux", "csigflux"]] = transformed[
            ["cfid", "cjd", "cflux", "csigflux"]
        ].apply(keep_filter, args=(band,), axis=1, result_type="expand")

        all_transformed.append(transformed)

    for df in all_transformed:

        df["cjd"] = df.apply(translate, axis=1)

        df["peak"] = df["cflux"].apply(get_max)
        df[["cflux", "csigflux"]] = df.apply(normalize, axis=1)
        df["snr"] = df[["cflux", "csigflux"]].apply(
            lambda pdf: pdf["cflux"] / pdf["csigflux"], axis=1
        )

    condition = []

    for pair in range(5):
        condition.append((all_transformed[pair]['cjd'].apply(len) >= minimum_points) & (all_transformed[pair + 1]['cjd'].apply(len) >= minimum_points))

    valid = np.array([False] * len(converted))
    for cond in condition:
        valid = valid | cond

    return [x[valid] for x in all_transformed], valid


def parametric_bump(ps):

    """Fit the lightcurves using the bump function. Extract the parameters
    Parameters
    ----------
    ps : pd.Series
        pd Series of alerts from Fink with nan removed and converted to flux.
        Flux must be normalised
        Lightcurves's max flux must be centered on 40.
        p4 guess is set to the minimum flux value
    Returns
    -------
    parameters : list
        List of best fitting parameter values [p1, p2, p3, p4]

    Examples
    --------

    >>> example = pd.Series(data = {"cjd" : np.array([-1, 0]),\
                            "cflux" : np.array([20, 300]),\
                            "csigflux" : np.array([1, 3]),\
                            "anything" : np.array(['toto'])})

    >>> np.array_equal(np.round(parametric_bump(example), 3), np.array([ 13.465, 119.56 , -14.035,  47.9  ]))
    True

    >>> example2 = pd.Series(data = {"cjd" : np.array([]),\
                            "cflux" : np.array([]),\
                            "csigflux" : np.array([]),\
                            "anything" : np.array(['toto'])})

    >>> np.array_equal(np.round(parametric_bump(example2), 3), np.array([ 0.225, -2.5  ,  0.038, -1.   ]))
    True
    """

    parameters_dict = {"p1": 0.225, "p2": -2.5, "p3": 0.038, "p4": get_min(ps["cflux"])}

    least_squares = LeastSquares(ps["cjd"], ps["cflux"], ps["csigflux"], mod.bump)
    fit = Minuit(least_squares, **parameters_dict)

    fit.migrad()

    parameters = []
    for fit_v in range(len(fit.values)):
        parameters.append(fit.values[fit_v])

    return parameters


def compute_color(ps, minimum=4):

    """Compute the color of an alert by computing g-r (before normalisation)
    Proceed by virtually filling missing points of each band using the bump fit

    Parameters
    ----------
    ps : pd.Series
        Dataframe of alerts from Fink with nan removed and converted to flux.
        Flux must be normalised with normalization factor inside column ['peak']
        Lightcurves's max flux must be centered on 40

    Returns
    -------
    np.array
        Array of color g-r at each point

    """

    all_colors = []

    for pair in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]:

        if (len(ps[f"cjd_{pair[1]}"]) >= minimum) & (
            len(ps[f"cjd_{pair[1]}"]) >= minimum
        ):

            # Compute fitted values at cjd from the other band
            add_from_1 = mod.bump(ps[f"cjd_{pair[1]}"], *ps[f"bump_{pair[0]}"])
            add_from_0 = mod.bump(ps[f"cjd_{pair[0]}"], *ps[f"bump_{pair[1]}"])

            # Add to the flux list : maintain the same order : cjd from 0 then cjd from 1
            new_cflux_0 = np.append(ps[f"cflux_{pair[0]}"], add_from_1)
            new_cflux_1 = np.append(add_from_0, ps[f"cflux_{pair[1]}"])

            unnorm_cflux_0 = new_cflux_0 * ps[f"peak_{pair[0]}"]
            unnorm_cflux_1 = new_cflux_1 * ps[f"peak_{pair[1]}"]

            all_colors.append(unnorm_cflux_0 - unnorm_cflux_1)

        else:
            all_colors.append([0])

    return all_colors


def compute_std(x):
    if len(x) == 0:
        return -1
    else:
        return np.std(x)


def compute_mean(x):
    if len(x) == 0:
        return -1
    else:
        return np.mean(x)


def parametrise(all_transformed, target_col=""):

    """Extract parameters from a transformed dataset. Construct a new DataFrame
    Parameters are :  - 'nb_points' : number of points
                      - 'std' : standard deviation of the flux
                      - 'peak' : maximum before normalization
                      - 'mean_snr' : mean signal over noise ratio

    Also compute a fit using the bump function on each lightcurve. Parameters
    of the fit will later be used to compute color parameters

    Parameters
    ----------
    transformed : pd.DataFrame
        Transformed DataFrame that only contains a single passband
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset
        Default is ''

    Returns
    -------
    df_parameters : pd.DataFrame
        DataFrame of parameters.
        Contains additionnal columns which will be used to compute color parameters

    """

    all_features = []

    for band in range(6):

        transformed = all_transformed[band]
        nb_points = transformed["cflux"].apply(lambda x: len(x))
        peak = transformed["peak"]
        std = transformed["cflux"].apply(compute_std)
        mean_snr = transformed["snr"].apply(compute_mean)
        ids = transformed["objectId"]
        ra = transformed["ra"]
        dec = transformed["dec"]
        hostgal_dist = transformed["hostgal_dist"]
        hostgal_zphot = transformed["hostgal_zphot"]
        hostgal_zphot_err = transformed["hostgal_zphot_err"]

        df_parameters = pd.DataFrame(
            data={
                "object_id": ids,
                "ra": ra,
                "dec": dec,
                "hostgal_dist": hostgal_dist,
                "hostgal_zphot": hostgal_zphot,
                "hostgal_zphot_err": hostgal_zphot_err,
                f"std_{band}": std,
                f"peak_{band}": peak,
                f"mean_snr_{band}": mean_snr,
                f"nb_points_{band}": nb_points,
            }
        )

        if target_col != "":
            targets = transformed[target_col]
            df_parameters["target"] = targets

        # Compute missing values for the color
        # The bump function is built to fit transient centered on 40
        transformed["cjd"] = transformed["cjd"].apply(lambda x: np.array(x) + 40)
        bump_parameters = transformed.apply(parametric_bump, axis=1)
        df_parameters[f"bump_{band}"] = bump_parameters

        df_parameters[f"cflux_{band}"] = transformed["cflux"]
        df_parameters[f"cjd_{band}"] = transformed["cjd"]

        all_features.append(df_parameters)

    return all_features


def merge_features(all_features, minimum_points, target_col=""):

    """Merge feature tables of band g and r.
    Compute color parameters : - 'max_color' : absolute maximum of the color
                               - 'std_color' : standard deviation of the color

    Parameters
    ----------
    features_1: pd.DataFrame
        features of band g
    features_2: pd.DataFrame
        features of band r
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset
        Default is ''

    Returns
    -------
    ordered_features : pd.DataFrame
        Final features dataset with ordered columns :
        ['object_id', 'std_1', 'std_2', 'peak_1', 'peak_2', 'mean_snr_1',
        'mean_snr_2', 'nb_points_1', 'nb_points_2', 'std_color', 'max_color']
    """

    # Avoid having twice the same column
    if target_col == "":
        for band in [1, 2, 3, 4, 5]:
            all_features[band] = all_features[band].drop(
                columns={
                    "object_id",
                    "ra",
                    "dec",
                    "hostgal_dist",
                    "hostgal_zphot",
                    "hostgal_zphot_err",
                }
            )
    else:
        for band in [1, 2, 3, 4, 5]:
            all_features[band] = all_features[band].drop(
                columns={
                    "object_id",
                    "ra",
                    "dec",
                    "hostgal_dist",
                    "hostgal_zphot",
                    "hostgal_zphot_err",
                    target_col,
                }
            )

    features = all_features[0]
    for band in [1, 2, 3, 4, 5]:
        features = features.join(all_features[band])

    ordered_features = features[
        [
            "object_id",
            "ra",
            "dec",
            "hostgal_dist",
            "hostgal_zphot",
            "hostgal_zphot_err",
            "std_0",
            "std_1",
            "std_2",
            "std_3",
            "std_4",
            "std_5",
            "peak_0",
            "peak_1",
            "peak_2",
            "peak_3",
            "peak_4",
            "peak_5",
            "mean_snr_0",
            "mean_snr_1",
            "mean_snr_2",
            "mean_snr_3",
            "mean_snr_4",
            "mean_snr_5",
            "nb_points_0",
            "nb_points_1",
            "nb_points_2",
            "nb_points_3",
            "nb_points_4",
            "nb_points_5",
        ]
    ].copy()

    # Add color features
    color = features.apply(compute_color, axis=1, args=(minimum_points,))
    color = color.apply(pd.Series)

    for idx, i in enumerate(["u-g", "g-r", "r-i", "i-z", "z-Y"]):

        ordered_features[f"std_{i}"] = color[idx].apply(compute_std)
        ordered_features[f"max_{i}"] = color[idx].apply(get_max, args=(True,))

    if target_col != "":
        ordered_features[target_col] = features[target_col]

    # Make sure that no value is above 2**32 (scipy uses float32)
    max_size = 2**30
    float_cols = ordered_features.columns[ordered_features.columns != 'object_id']

    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] > max_size, max_size)
    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] < -max_size, -max_size)

    return ordered_features


def get_probabilities(clf, features, valid):

    """Returns probabilty of being an AGN predicted by the classfier

    Parameters
    ----------
    clf: RandomForestClassifier
        Binary AGN vs non AGN classifier

    features: pd.DataFrame
        Features extracted from the objects.
        Outputed by merge_features

    valid: np.array
        Bool array, indicates if both passband respect the minimum number of points

    Returns
    -------
    final_proba : np.array
        ordered probabilities of being an AGN
    """

    final_proba = np.array([0.0] * len(valid)).astype(np.float64)

    if len(features) > 0:
        features = features.replace(np.inf, 0.0).replace(np.nan, 0.0)
        agn_or_not = clf.predict_proba(features.iloc[:, 1:])
        index_to_replace = features.iloc[:, 1:].index
        final_proba[index_to_replace.values] = agn_or_not[:, 1]

    return final_proba


if __name__ == "__main__":

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
