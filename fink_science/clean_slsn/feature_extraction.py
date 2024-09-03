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

import pickle  # noqa: F401
import warnings
from scipy.optimize import curve_fit

import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

import fink_science.clean_slsn.kernel as k
import fink_science.clean_slsn.models as mod
import fink_science.clean_slsn.basic_functions as base

from pandas.testing import assert_frame_equal  # noqa: F401


@profile
def transform_data(data):
    """Apply transformations for each filters on a flux dataset
            - Shift cmidPointTai so that the max flux point is at 0
            - Normalize by dividing flux and flux err by the
            maximum flux of the k.NORMALIZING_BAND (kernel option)
            - Add a column with maxflux before normalization

    Split the results into multiple dataframes each containing only one passband.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of alerts.

    Returns
    -------
    all_transformed : list
        List of DataFrame. Each df is a transformed version of data
        that only contains observations from one passband and valid objects.
    valid: np.array
        Boolean array describing if each object is valid.
        Objects are valid if all required band have at least k.MINIMUM_POINTS observations.

    Examples
    --------
    """

    peak = data['cpsFlux'].apply(base.get_max)
    
    all_transformed = []
    valid = np.array([True] * len(data))

    for band in k.PASSBANDS:
        transformed = data.copy()
        transformed[["cfilterName", "cmidPointTai", "cpsFlux", "cpsFluxErr"]] = transformed[
            ["cfilterName", "cmidPointTai", "cpsFlux", "cpsFluxErr"]
        ].apply(base.keep_filter, args=(band,), axis=1, result_type="expand")

        all_transformed.append(transformed)

        if band in k.COLOR_PAIR:
            valid = valid & (transformed["cmidPointTai"].apply(lambda x: len(x) >= k.MINIMUM_POINTS))

    all_transformed = [x[valid].copy() for x in all_transformed]

    if not valid.any():
        return all_transformed, valid

    for df in all_transformed:
        df['peak'] = peak
        df["cmidPointTai"] = df.apply(base.translate, axis=1)
        df[["cpsFlux", "cpsFluxErr"]] = df.apply(base.normalize, axis=1)
        df["snr"] = df[["cpsFlux", "cpsFluxErr"]].apply(
            lambda pdf: pdf["cpsFlux"] / pdf["cpsFluxErr"], axis=1
        )

    return all_transformed, valid

@profile
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
        fit = curve_fit(mod.mvsr_right_transient, ps[f"cmidPointTai_{band}"], ps[f"cpsFlux_{band}"], sigma=ps[f"cpsFluxErr_{band}"], p0=[-0.005, 0.015, 15], bounds=([-2, 0, -300], [0, 2, 300]), maxfev=k.MAXFEV)

    except (RuntimeError, ValueError, LinAlgError):
        fit = [[0, 0, 0]]

    return fit[0]


@profile
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

    # Compute fitted values at cmidPointTai from the other band
    add_from_1 = mod.mvsr_right_transient(ps["cmidPointTai_red"], *ps["func_blue"])
    add_from_0 = mod.mvsr_right_transient(ps["cmidPointTai_blue"], *ps["func_red"])

    # Add to the flux list : maintain the same order : cmidPointTai from 0 then cmidPointTai from 1
    new_cpsFlux_0 = np.append(ps["cpsFlux_blue"], add_from_1)
    new_cpsFlux_1 = np.append(add_from_0, ps["cpsFlux_red"])

    return (new_cpsFlux_0 - new_cpsFlux_1) * ps["peak"]


@profile
def parametrise(all_transformed, metadata, target_col=""):
    """Extract parameters from a list of dataset outputed
       by the transform_data function.

    Parameters are :
            - "peak" : maximum flux before normalization for filter k.NORMALIZING_BAND
            - "ra" : right ascension
            - "decl" : declination
            
            Optional metadata:
                - "hostgal_snsep" : distance to host galaxy
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
        Also adds columns of cmidPointTai, cpsFlux and cpsFluxErr that
        will be used to compute color later on.

    Example
    -------
    """

    all_features = []

    for idx, band in enumerate(k.PASSBANDS):

        transformed = all_transformed[idx]
        nb_points = transformed["cpsFlux"].apply(lambda x: len(x))
        peak = transformed["peak"]
        std = transformed["cpsFlux"].apply(base.compute_std)
        mean_snr = transformed["snr"].apply(base.compute_mean)
        ids = transformed["diaObjectId"]
        ra = transformed["ra"]
        decl = transformed["decl"]
        
        if metadata:
            hostgal_snsep = transformed["hostgal_snsep"]
            hostgal_zphot = transformed["hostgal_zphot"]
            hostgal_zphot_err = transformed["hostgal_zphot_err"]

            df_parameters = pd.DataFrame(
                data={
                    "object_id": ids,
                    "ra": ra,
                    "decl": decl,
                    "hostgal_snsep": hostgal_snsep,
                    "hostgal_zphot": hostgal_zphot,
                    "hostgal_zphot_err": hostgal_zphot_err,
                    "peak": peak,
                    f"std_{band}": std,
                    f"mean_snr_{band}": mean_snr,
                    f"nb_points_{band}": nb_points,
                }
            )
        
        else:
            df_parameters = pd.DataFrame(
                data={
                    "object_id": ids,
                    "ra": ra,
                    "decl": decl,
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
            df_parameters[f"cpsFlux_{band}"] = transformed["cpsFlux"]
            df_parameters[f"cpsFluxErr_{band}"] = transformed["cpsFluxErr"]
            df_parameters[f"cmidPointTai_{band}"] = transformed["cmidPointTai"]

        all_features.append(df_parameters)

    return all_features


@profile
def merge_features(all_features, metadata, target_col=""):
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

    features = all_features[0]
    singular_features = ["object_id","ra","decl","peak"]
    if metadata:
        singular_features += ["hostgal_snsep","hostgal_zphot","hostgal_zphot_err"]

    # Avoid having twice the same column

    for band in range(1, len(k.PASSBANDS)):
        all_features[band] = all_features[band].drop(
            columns={
                *singular_features,
                target_col,
            }, errors='ignore')

        features = features.join(all_features[band])

    ordered_features = features[
        [
            *singular_features,
            *[f"std_{i}" for i in k.PASSBANDS],
            *[f"mean_snr_{i}" for i in k.PASSBANDS],
            *[f"nb_points_{i}" for i in k.PASSBANDS]
        ]
    ].copy()

    features[['func_blue', 'cmidPointTai_blue', 'cpsFlux_blue', 'chi2_blue'
              'func_red', 'cmidPointTai_red', 'cpsFlux_red', 'chi2_red']] = None

    pair = k.COLOR_PAIR

    for colo_idx, colo in enumerate(['blue', 'red']):

        features[f'func_{colo}'] = features.apply(parametric_func, axis=1, args=(pair[colo_idx],))
        for colname in ['cmidPointTai', 'cpsFlux', 'cpsFluxErr']:
            features[f'{colname}_{colo}'] = features[f'{colname}_{pair[colo_idx]}']

        for p in range(3):
            ordered_features[f'p{p+1}_{colo}'] = features[f'func_{colo}'].apply(lambda x: x[p])

        ordered_features[f'chi2_{colo}'] = features.apply(compute_chi2, axis=1, args=(colo,))

    # Add color features
    features['color'] = features.apply(compute_color, axis=1)

    ordered_features["std_color"] = features['color'].apply(base.compute_std)
    ordered_features["max_color"] = features['color'].apply(base.get_max, args=(True,))

    # Make sure that no value is above 2**32 (scipy uses float32)
    max_size = 2**30
    float_cols = ordered_features.columns[ordered_features.columns != 'object_id']

    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] > max_size, max_size)
    ordered_features[float_cols] = ordered_features[float_cols].mask(ordered_features[float_cols] < -max_size, -max_size)

    if target_col != "":
        targets = features[target_col]
        ordered_features[target_col] = targets

    return ordered_features


@profile
def compute_chi2(pdf, color):

    x = pdf[f'cmidPointTai_{color}']
    y = pdf[f'cpsFlux_{color}']
    yerr = pdf[f'cpsFluxErr_{color}']
    parameters = pdf[f'func_{color}']

    return np.sum((y - mod.mvsr_right_transient(x, *parameters))**2 / yerr)


if __name__ == "__main__":

    import sys
    import doctest

    sys.exit(doctest.testmod()[0])
