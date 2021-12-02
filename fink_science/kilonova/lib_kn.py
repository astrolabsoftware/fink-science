# Copyright 2021 Fink Software
# Author: Emille E. O. Ishida
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
import numpy as np
from scipy.optimize import minimize

def get_features_name(npcs):
    """
    Create the list of feature names depending on the number of principal components.

    Parameters
    ----------
    npcs : int
        number of principal components to use

    Returns
    -------
    list
        name of the features.

    """
    names_root = [
        'coeff' + str(i + 1) + '_' for i in range(npcs)] + [
            'residuo_',
            'maxflux_']

    return [i + j for j in ['g', 'r'] for i in names_root]


def calc_prediction(coeff, pcs_arr):
    """
    given the coefficients and PCs, it calculates the prediction as a linear combination

    Parameters
    ----------
    coeff: np.array of shape [num_pcs]
        coefficients of the linear combinations for the PCs
    pcs_arr: np.array of shape [num_pcs, num_prediction_points]
        The PCs that are being used as templates

    Returns
    -------
    predicted_lc: np.array of shape [num_prediction_points]
        prediction as a linear comination of PCs
    """
    predicted_lc = np.zeros_like(pcs_arr.shape[0])
    for a, b in zip(pcs_arr, coeff):
        predicted_lc = np.add(predicted_lc, b * a)

    return predicted_lc


def calc_loss(coeff, pcs_arr, light_curve_flux, light_curve_err, map_dates_to_arr_index, regularization_weight, low_var_indices=[1]):
    """
    function to calculate the loss to be optimized

    Parameters
    ----------
    coeff: np.array of shape [num_of_pcs]
        current value of coefficients
    pcs_arr: np.array of shape [num_pcs, num_prediction_points]
        principal components to the used for the prediction
    light_curve_flux: pandas column of shape [num_recorded_points]
        segment of lightcurve that is to be fitted
    light_curve_err: pandas column of shape [num_recorded_points]
        segment with corresponding error bars in the segment that is to be fitted.
    map_dates_to_arr_index: np.array of shape [num_recorded_points]
        maping that holds the index position corresponding to each point in the lightcurve
    regularization_weight: float
        weights given to the regularization term
    low_var_indices: list
        Indices along which variance is low.
        Default value is set to [1] which regularizes the 2nd PC

    Returns
    -------
    loss: (float)
        that is to be optimized
    """
    # calculation of the reconstruction loss
    y_pred = calc_prediction(coeff, pcs_arr)
    real_flux = np.take(y_pred, map_dates_to_arr_index)
    reconstruction_loss = np.sum(np.divide(np.square(real_flux - light_curve_flux), np.square(light_curve_err)))

    # Calculate the regularization

    # Regularize the second coefficient
    regularization_term = 0
    if low_var_indices is not None:
        regularization_term = np.sum(np.abs(coeff[low_var_indices[:]]))

    # Regularize negative pcs
    neg_coeff = coeff[coeff < 0]
    regularization_term = regularization_term + np.sum(np.abs(neg_coeff))

    loss = reconstruction_loss + regularization_term * regularization_weight

    return loss


def calc_residual(coeff, pcs_arr, light_curve_flux, light_curve_err, map_dates_to_arr_index):
    """
    function to calculate residual of the fit

    Parameters
    ----------
    coeff: np.array of shape [num_of_pcs]
        current value of coefficients
    pcs_arr: np.array of shape [num_pcs, num_prediction_points]
        principal components to the used for the prediction
    light_curve_flux: pandas column of shape [num_recorded_points]
        segment of lightcurve that is to be fitted
    light_curve_err: pandas column of shape [num_recorded_points]
        segment with corresponding error bars in the segment that is to be fitted.
    map_dates_to_arr_index: np.array of shape [num_recorded_points]
        maping that holds the index position corresponding to each point in the lightcurve

    Returns
    -------
    residual: float
        residual value
    """

    y_pred = calc_prediction(coeff, pcs_arr)
    real_flux = np.take(y_pred, map_dates_to_arr_index)

    diff = real_flux - light_curve_flux
    reconstruction_loss = np.mean(np.divide(np.square(diff), np.square(light_curve_err)))

    residual = np.sqrt(reconstruction_loss)
    return residual

def predict_band_features(band_df, pcs_arr, time_bin=.25, flux_lim=200, low_var_indices=[1]):
    """
    function to evaluate features for a band

    Parameters
    ----------
    band_df: pandas.DataFrame
        dataframe with the data of only one band of a lightcurve
    pcs_arr: np.array of shape [num pc components, num prediction points/bins]
        For example, pcs_arr[0] will correspond the the first principal component.
    time_bin: float
        Width of time gap between two elements in PCs.
    flux_lim: float (optional)
        Limit of minimum flux for prediction to be made in a band.
        Note that all the points in the band is used for the fit provided that max flux in the band > flux_lim
    low_var_indices: list
        Indices along which variance is low.
        Default value is set to [1] which regularizes the 2nd PC

    Returns
    -------
    features: list of features for the given band
        The features are in the same order in which the classifier was trained:
        coefficients of pcs, number of features, residual and maxflux.
    """

    num_pcs = len(pcs_arr)
    num_prediction_points = len(pcs_arr[0])

    if len(band_df) == 0:
        features = np.zeros(int(len(get_features_name(num_pcs)) / 2)).tolist()
        return features

    max_loc = np.argmax(band_df['FLUXCAL'])
    max_flux = band_df['FLUXCAL'].iloc[max_loc]

    # extract the prediction region
    mid_point_date = band_df['MJD'].iloc[max_loc]

    prediction_duration = time_bin * (num_prediction_points - 1)

    start_date = mid_point_date - prediction_duration / 2
    end_date = mid_point_date + prediction_duration / 2

    duration_index = (band_df['MJD'] > start_date) & (band_df['MJD'] < end_date)
    band_df = band_df[duration_index]

    if (max_flux > flux_lim) & (len(band_df) > 2):

        # update the location
        max_loc = np.argmax(band_df['FLUXCAL'])

        # create a mapping from JD to index in the prediction.
        # For Example, midpoint is at index (num_prediction_points - 1) / 2. The middle of the prediction region.
        map_dates_to_arr_index = np.around((band_df['MJD'] - mid_point_date).to_numpy().astype(float) / time_bin + (num_prediction_points - 1) / 2)
        map_dates_to_arr_index = map_dates_to_arr_index.astype(int)

        # Initil guess for coefficients.
        initial_guess = np.zeros(num_pcs) + 0.5

        # Calculating the regularization weight to make it comparable to reconstruction loss part.
        err_bar_of_max_flux = band_df['FLUXCALERR'].iloc[max_loc]

        regularization_weight = np.square(max_flux / err_bar_of_max_flux)

        # normalize the flux and errorbars
        normalized_flux = band_df['FLUXCAL'].to_numpy() / max_flux
        normalized_err_bars = band_df['FLUXCALERR'].to_numpy() / max_flux

        # bounds for the coefficient
        bounds = []
        for i in range(num_pcs):
            bounds.append([-2, 2])

        # minimize the cost function
        result = minimize(
            calc_loss, initial_guess,
            args=(
                pcs_arr, normalized_flux, normalized_err_bars,
                map_dates_to_arr_index, regularization_weight, low_var_indices),
            bounds=bounds)

        # extract the coefficients
        coeff = list(result.x)

        # maximum flux in a band
        max_band_flux = max_flux

        # calculate residuals
        residual = calc_residual(result.x, pcs_arr, normalized_flux, normalized_err_bars, map_dates_to_arr_index)

    else:
        coeff = np.zeros(num_pcs).tolist()
        residual = 0
        max_band_flux = 0

    # buid features list
    features = coeff
    features.append(residual)
    features.append(max_band_flux)

    return features

def extract_all_filters_fink(pcs, time_bin, filters, lc, flux_lim=200, low_var_indices=[1]):
    """
    Extract features for all the bands of lightcurve
    Parameters
    ----------
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...),
        values their amplitude at each epoch in the grid.
        Order of PCs when calling pcs.keys() is important.
    time_bin: float
        Width of time gap between two elements in PCs.
    filters: list
        List of broad band filters.
    lc: pd.DataFrame
        Keys should be ['MJD', 'FLUXCAL', 'FLUXCALERR', 'FLT'].
    flux_lim: float (optional)
        Limit of minimum flux for prediction to be made in a band.
        Note that all the points in the band is used for the fit provided that max flux in the band > flux_lim
    low_var_indices: list
        Indices along which variance is low.
        Default value is set to [1] which regularizes the 2nd PC

    Returns
    -------
    all_features: list
        List of features for this object.
        Order is all features from first filter, then all features from
        second filters, etc.
    """

    pcs_arr = []
    for i in range(len(pcs.keys())):
        pcs_arr.append(pcs[i + 1].values)
    pcs_arr = np.array(pcs_arr)
    all_features = []

    for band in filters:

        band_df = lc[lc['FLT'] == band]
        features = predict_band_features(
            band_df=band_df, pcs_arr=pcs_arr, time_bin=time_bin,
            flux_lim=flux_lim, low_var_indices=low_var_indices)

        all_features.extend(features)

    return all_features
