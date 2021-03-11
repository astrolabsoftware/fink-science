# Copyright 2020 AstroLab Software
# Author: Marco Leoni, Julien Peloton
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
import numpy as np
import random

from fink_science.random_forest_snia.sigmoid import fit_sigmoid
from fink_science.random_forest_snia.sigmoid import delta_t
from fink_science.random_forest_snia.sigmoid import compute_chi_square
from fink_science.random_forest_snia.sigmoid import fsigmoid

columns_to_keep = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
fluxes = ['FLUXCAL', 'FLUXCALERR']
RF_FEATURE_NAMES = 'a_g,b_g,c_g,snratio_g,chisq_g,nrise_g,a_r,b_r,c_r,snratio_r,chisq_r,nrise_r'.split(',')

def filter_data(data, filt):
    """Select data according to the value of the
       filter (for ZTF only g, r)

        Parameters
        ----------
        data: pandas DataFrame
            containing light curves for different filters
        filt: string
            defining the filter ('g' or 'r')

         Returns
         -------
         data_filt: pandas DataFrame
        containing light curves for the selected filter, filt

        """

    data_filt = data.loc[data['FLT'] == filt]

    return data_filt


def mask_negative_data(data, low_bound):
    """Mask data points whose FLUXCAL values are
       lower than a chosen lower bound

       Prameteres
       ----------
       data: pandas DataFrame
       light curve data for given filter
       low_bound: float
       minimum allowed value of flux

       Returns
       -------
       data: pandas DataFrame
       light curve with masked flux

        """

    masked_data = data.mask(data['FLUXCAL'] < low_bound)\
        .set_index(data['MJD'])[fluxes].dropna()
    return masked_data


def get_fake_df(filt):
    """Get fake data frame

    Parameters
    ----------
    filt: char
    name of the desired filter

    Returns
    ------
    fake: pandas DataFrame with fake values

    """

    fake = pd.DataFrame({
        'VARLIST:': np.array(['OBS:', 'OBS:', 'OBS:']),
        'MJD': np.array([1, 2, 3]),
        'FLT': np.array([filt, filt, filt]),
        'FIELD': np.array([0.0, 0.0, 0.0]),
        'FLUXCAL': np.array([0, 0, 0]),
        'FLUXCALERR': np.array([1, 1, 1])
    })

    return fake


def get_fake_fit_parameters():
    """Function returning fake values
    for the fit parameters

    Returns
    -------
    [a, b, c, snratio, chisq] = list of floats
    list of fake features in the absence of real fit

    """

    [a, b, c, snratio, chisq] = [0, 0, 0, 0.1, 1e8]
    # [a, b, c, snratio, chisq] = [0, 0, 0, 0.0, 0.0]

    return [a, b, c, snratio, chisq]


def get_fake_results(filt):
    """Returns fake results in case of problems with the fit

    Parameters
    ----------
    filt:char, the filter name (g, r)

    Returns
    ------
    [export, a, b, c, snratio, chisq, nrise]: list of
        DataFrame and floats
    list of fake results in the absence of a real fit

    """
    nrise = 0
    [a, b, c, snratio, chisq] = get_fake_fit_parameters()

    return [a, b, c, snratio, chisq, nrise]


def get_ewma_derivative(data, ewma_window):
    """Compute the ewma and the derivative

    Parameters
    ----------
    data: pandas DataFrame
    ewma_window: width of the window
    in the rolling average

    Return
    ------
    ewma_derivative: pandas DataFrame
    derivative of the ewma of data

    """

    ewma_derivative = data.ewm(ewma_window, min_periods=0).mean().diff()

    return ewma_derivative


def get_sn_ratio(data, data_err):
    """Compute signal to noise ratio

    Parameters
    ----------
    data: np.array
     rising flux, 'FLUXCAL'
    data_err: np.array
     error in the rising flyx, 'FLUXCALERR'

    Returns
    -------
    snr: float
    the average signal-to-noise ratio
    in the considered interval

    """

    # average signal to noise ratio
    snr = (data / data_err).mean()

    return snr


def get_predicted_flux(dt, a, b, c):
    """Compute the expected flux, in the interval dt,
     using fitted parameters a, b ,c

    Parameters
    ----------
    dt: np.array
    time in days (relative to the initial time t0)
    a:float
    parameter (timescale) of the sigmoid function
    b:float
    parameter (temporal) of the sigmoid function
    c:float
    parameter (height) of the sigmoid function

    Returns
    -------
    predicted_flux: np.array
    with predicted flux based on the fitted values a, b, c

    """

    predicted_array = fsigmoid(dt, a, b, c)
    predicted_flux = pd.DataFrame(predicted_array)\
        .round(decimals=4)\
        .replace(0.0, 0.00001)[0]\
        .values

    return predicted_flux


def get_data_to_export(data_full, data_rising):
    """Export new set of data containing contigous
    rising data

    Parameters
    ----------
    data_full: pandas DataFrame
    full dataset, containing flux, 'MJD' and other labels
    data_rising: pandas DataFrame
    rising dataset, containing flux only

    Returns
    -------
    to_export: pandas DataFrame
    rising dataset, containing flux, 'MJD' and other labels

    """

    to_export = data_full.loc[
        data_full['MJD'].isin(data_rising.index.values)].dropna()

    # set field to zero
    to_export['FIELD'] = 0

    return to_export


def get_train_test(percent_train):
    """Randomly choose test or train label

    Parameters
    ----------
    percent_train: float
    the fraction of the training set
    with respect to the full (training + test) set

    Returns
    -------
    sample: string
    'test' or 'train' set

    """

    rando = random.uniform(0, 1)

    if(rando < percent_train):
        sample = 'train'
    else:
        sample = 'test'

    return sample


def average_intraday_data(df_intra):
    """Average over intraday data points

     Parameters
     ----------
     df_intra: pd.DataFrame
        containing the history of the flux
        with intraday data

     Returns
     -------
     df_average: pd.DataFrame
        containing only daily data
    """

    df_average = df_intra.copy()
    df_average['MJD'] = df_average['MJD'].apply(
        lambda x: np.around(x, decimals=0))
    df_average = df_average.groupby('MJD').mean()
    df_average['MJD'] = df_average.index.values

    return df_average


def get_sigmoid_features_dev(data_all: pd.DataFrame):
    """Compute the features needed for the Random Forest classification based
    on the sigmoid model.

    Parameters
    ----------
    data_all: pd.DataFrame
        Pandas DataFrame with at least ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR']
        as columns.

    Returns
    -------
    out: list of floats
        List of features, ordered by filter bands:
        [a['g'], b['g'], c['g'], snratio['g'], chisq['g'], nrise['g'],
         a['r'], b['r'], c['r'], snratio['r'], chisq['r'], nrise['r']]

    """
    # lower bound on flux
    low_bound = -10

    # width of the ewma window
    ewma_window = 3

    # N min data points
    min_data_points = 4

    # N min rising data points
    min_rising_points = 3

    list_filters = ['g', 'r']

    # features for different filters
    a = {}
    b = {}
    c = {}
    snratio = {}
    chisq = {}
    nrise = {}

    for i in list_filters:
        # select filter
        data_tmp = filter_data(data_all[columns_to_keep], i)
        # average over intraday data points
        data_tmp_avg = average_intraday_data(data_tmp)
        # mask negative flux below low bound
        data_mjd = mask_negative_data(data_tmp_avg, low_bound)

        # check data have at least 5 points
        if len(data_mjd['FLUXCAL'].values > min_data_points):
            # compute the derivative
            deriv_ewma = get_ewma_derivative(data_mjd['FLUXCAL'], ewma_window)
            # mask data with negative part
            data_masked = data_mjd.mask(deriv_ewma < 0)
            # get longest raising sequence
            rising_data = data_masked.dropna()

            # at least three points (needed for the sigmoid fit)
            if(len(rising_data) > min_rising_points):

                # focus on flux
                rising_time = rising_data['FLUXCAL'].index.values
                rising_flux = rising_data['FLUXCAL'].values
                rising_flux_err = rising_data['FLUXCALERR'].values

                # compute signal to noise ratio
                snratio[i] = get_sn_ratio(rising_flux, rising_flux_err)

                # get N rising points
                nrise[i] = len(rising_flux)

                dt = delta_t(rising_time)

                # perform sigmoid fit
                [a[i], b[i], c[i]] = fit_sigmoid(dt, rising_flux)

                # perform sigmoid fit
                [a[i], b[i], c[i]] = fit_sigmoid(dt, rising_flux)

                # predicted flux with fit parameters
                pred_flux = get_predicted_flux(dt, a[i], b[i], c[i])

                # compute chi-square
                chisq[i] = compute_chi_square(rising_flux, pred_flux)

            else:
                # if rising flux has less than three
                [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                    get_fake_results(i)
        else:
            # if data points not enough
            [a[i], b[i], c[i], snratio[i], chisq[i], nrise[i]] = \
                get_fake_results(i)

    return [
        a['g'], b['g'], c['g'], snratio['g'], chisq['g'], nrise['g'],
        a['r'], b['r'], c['r'], snratio['r'], chisq['r'], nrise['r']
    ]
