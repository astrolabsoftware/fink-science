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
import fink_science.pisn.models as mod
import fink_science.agn.feature_extraction as fe_agn
from pandas.testing import assert_frame_equal  # noqa: F401
import fink_science.pisn.kernel as k
import numpy as np
import pickle  # noqa: F401
from scipy.optimize import curve_fit
import warnings


def transform_data(formated):
    """Apply transformations for each filters on a flux formated dataset
            - Shift cjd so that the max flux point is at 0
            - Normalize by dividing flux and flux err by the
            maximum flux of the k.NORMALIZING_BAND (kernel option)
            - Add a column with maxflux before normalization

    Split the results into multiple dataframes each containing only one passband.

    Parameters
    ----------
    formated : pd.DataFrame
        Dataframe of alerts formated using "format_data" function.

    Returns
    -------
    all_transformed : list
        List of DataFrame. Each df is a transformed version of formated
        that only contains observations from one passband and valid objects.
    valid: np.array
        Boolean array describing if each object is valid.
        Objects are valid if all required band have at least k.MINIMUM_POINTS observations.

    Examples
    --------
    """

    all_transformed = []
    passbands = [0, 1, 2, 3, 4, 5]
    valid = np.array([True] * len(formated))

    for band in passbands:
        transformed = formated.copy()
        transformed[["cfid", "cjd", "cflux", "csigflux"]] = transformed[
            ["cfid", "cjd", "cflux", "csigflux"]
        ].apply(fe_agn.keep_filter, args=(band,), axis=1, result_type="expand")

        all_transformed.append(transformed)

        if band in k.COLOR_PAIR:
            valid = valid & (transformed["cjd"].apply(lambda x: len(x) >= k.MINIMUM_POINTS))

    all_transformed = [x[valid].copy() for x in all_transformed]

    if not valid.any():
        return all_transformed, valid

    peak = all_transformed[k.NORMALIZING_BAND]['cflux'].apply(fe_agn.get_max)

    for df in all_transformed:
        df['peak'] = peak
        df["cjd"] = df.apply(fe_agn.translate, axis=1)
        df[["cflux", "csigflux"]] = df.apply(fe_agn.normalize, axis=1)
        df["snr"] = df[["cflux", "csigflux"]].apply(
            lambda pdf: pdf["cflux"] / pdf["csigflux"], axis=1
        )

    return all_transformed, valid


def parametric_func(ps, band):

    """Fit the lightcurves using the mvsr transient function.
    Extract the minimized parameters of the fit.

    Parameters
    ----------
    ps: pd.Series
        Alerts that have been transformed using 'transform_data' function.
    band: int
        Integer associated with the filter to fit.

    Returns
    -------
    list
        List of best fitting parameter values [p1, p2, p3]
        Returns [[0, 0, 0]] if the fit didn't converge.

    Examples
    --------
    """

    try:
        fit = curve_fit(mod.mvsr_right_transient, ps[f"cjd_{band}"], ps[f"cflux_{band}"], sigma=ps[f"csigflux_{band}"], p0=[-0.005, 0.015, 15], bounds=([-10, 0, 0], [0, 200, 500]), maxfev=k.MAXFEV)

    except (RuntimeError, ValueError):
        fit = [[0, 0, 0]]

    return fit[0]


def compute_color(ps):
    """Compute the color of an alert by computing blue-red
    Proceed by virtually filling missing points of each band using the mvsr transient fit

    Parameters
    ----------
    ps: pd.Series
        Dataframe of alerts as outputed by the parametrise function.

    Returns
    -------
    np.array
        Array of color blue-red at each point

    Examples
    --------
    """

    # Compute fitted values at cjd from the other band
    add_from_1 = mod.mvsr_right_transient(ps["cjd_red"], *ps["func_blue"])
    add_from_0 = mod.mvsr_right_transient(ps["cjd_blue"], *ps["func_red"])

    # Add to the flux list : maintain the same order : cjd from 0 then cjd from 1
    new_cflux_0 = np.append(ps["cflux_blue"], add_from_1)
    new_cflux_1 = np.append(add_from_0, ps["cflux_red"])

    return (new_cflux_0 - new_cflux_1) * ps["peak"]


def parametrise(all_transformed, target_col=""):
    """Extract parameters from a list of dataset outputed
       by the transform_data function.

    Parameters are :
            - "peak" : maximum flux before normalization for filter k.NORMALIZING_BAND
            - "ra" : right ascension
            - "dec" : declination
            - "hostgal_dist" : distance to host galaxy
            - "hostgal_zphot" : redshift of the host galaxy
            - "hostgal_zphot_err" : error on the redshift of the host galaxy

            For each filter:
                - 'std' : standard deviation of the flux for each filter
                - 'mean_snr' : mean signal over noise ratio for each filter
                - 'nb_points' : number of points for each filter

    Parameters
    ----------
    all_transformed : list
        List of transformed DataFrame using "transform_data" function.
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset.
        Default is ''

    Returns
    -------
    df_parameters : pd.DataFrame
        DataFrame of parameters.
        Also adds columns of cjd, cflux and csigflux that
        will be used to compute color later on.

    Example
    -------
    """
    passbands = [0, 1, 2, 3, 4, 5]

    all_features = []

    for idx, band in enumerate(passbands):

        transformed = all_transformed[idx]
        nb_points = transformed["cflux"].apply(lambda x: len(x))
        peak = transformed["peak"]
        std = transformed["cflux"].apply(fe_agn.compute_std)
        mean_snr = transformed["snr"].apply(fe_agn.compute_mean)
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
                "peak": peak,
                f"std_{band}": std,
                f"mean_snr_{band}": mean_snr,
                f"nb_points_{band}": nb_points,
            }
        )

        if target_col != "":
            targets = transformed[target_col]
            df_parameters[target_col] = targets

        if band in k.COLOR_PAIR:
            df_parameters[f"cflux_{band}"] = transformed["cflux"]
            df_parameters[f"csigflux_{band}"] = transformed["csigflux"]
            df_parameters[f"cjd_{band}"] = transformed["cjd"]

        all_features.append(df_parameters)

    return all_features


def merge_features(all_features, target_col=""):
    """Merge feature tables of all filters.
    Additionnaly fit requested bands and add fitted values as parameters:
    Using the fits, it computes color parameters :
                                - 'max_color' : absolute maximum of the color
                                - 'std_color' : standard deviation of the color

    Parameters
    ----------
    all_features: DataFrame
        Parameter dataframe, output of the "parametrise" function.
    target_col: str
        If inputed a non empty str, add the corresponding
        column as a target column to the final dataset.
        Default is ''.

    Returns
    -------


    Example
    -------
    """

    warnings.filterwarnings('ignore', '.*Covariance of the parameters could not be estimated.*')

    passbands = [0, 1, 2, 3, 4, 5]

    features = all_features[0]

    # Avoid having twice the same column

    for band in range(1, len(passbands)):
        all_features[band] = all_features[band].drop(
            columns={
                "object_id",
                "ra",
                "dec",
                "hostgal_dist",
                "hostgal_zphot",
                "hostgal_zphot_err",
                "peak",
                target_col,
            }, errors='ignore')

        features = features.join(all_features[band])

    ordered_features = features[
        [
            "object_id",
            "ra",
            "dec",
            "hostgal_dist",
            "hostgal_zphot",
            "hostgal_zphot_err",
            "peak",
            "std_0",
            "std_1",
            "std_2",
            "std_3",
            "std_4",
            "std_5",
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

    features[['func_blue', 'cjd_blue', 'cflux_blue', 'chi2_blue'
              'func_red', 'cjd_red', 'cflux_red', 'chi2_red']] = None

    pair = k.COLOR_PAIR

    for colo_idx, colo in enumerate(['blue', 'red']):

        features[f'func_{colo}'] = features.apply(parametric_func, axis=1, args=(pair[colo_idx],))
        for colname in ['cjd', 'cflux', 'csigflux']:
            features[f'{colname}_{colo}'] = features[f'{colname}_{pair[colo_idx]}']

        for p in range(3):
            ordered_features[f'p{p+1}_{colo}'] = features[f'func_{colo}'].apply(lambda x: x[p])

        ordered_features[f'chi2_{colo}'] = features.apply(compute_chi2, axis=1, args=(colo,))

    # Add color features
    features['color'] = features.apply(compute_color, axis=1)

    ordered_features["std_color"] = features['color'].apply(fe_agn.compute_std)
    ordered_features["max_color"] = features['color'].apply(fe_agn.get_max, args=(True,))

    # Make sure that no value is above 2**32 (scipy uses float32)
    max_size = 2**30
    float_cols = ordered_features.columns[ordered_features.columns != 'object_id']

    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] > max_size, max_size)
    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] < -max_size, -max_size)

    if target_col != "":
        targets = features[target_col]
        ordered_features[target_col] = targets

    return ordered_features


def compute_chi2(pdf, color):

    x = pdf[f'cjd_{color}']
    y = pdf[f'cflux_{color}']
    yerr = pdf[f'csigflux_{color}']
    parameters = pdf[f'func_{color}']

    return np.sum((y - mod.mvsr_right_transient(x, *parameters))**2 / yerr)


if __name__ == "__main__":

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
