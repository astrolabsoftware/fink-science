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
import pandas as pd

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
        'npoints_',
        'residuo_'
    ] + [
        'coeff' + str(i + 1) + '_' for i in range(npcs)
    ] + ['maxflux_']

    return [i + j for j in ['g', 'r'] for i in names_root]


def filter_points(
        obs_mjd: np.array, obs_flux: np.array,
        pc_epoch_grid: np.array):
    """Translate observed points to an epoch grid to match the PCs.

    Parameters
    ----------
    obs_mjds: np.array
        Values for observed mjds.
    obs_flux: np.array
        Values for fluxes at observed mjds.
    pc_epoch_grid: np.array
        Values of epochs grid used in constructing the PCs.
        Time bin between each entry should be the same.

    Returns
    -------
    new_mjd: np.array
        Values of mjds compatible to observations and PCs.
    new_flux: np.array
        Values of flux for each new_mjd.
        If more than one observation is available in a time bin
        this corresponds to the mean of all observations within
        the bin.
    mjd_flag: np.array of bool
        Mask for pc_epoch_grid filtering only allowed MJDs.
    mjd_cent: float
        Centered MJD value.
    """

    flux_final = []
    mjd_flag = []

    # get time bin
    time_bins = [np.round(pc_epoch_grid[i + 1] - pc_epoch_grid[i], 3)
                 for i in range(1, len(pc_epoch_grid) - 1)]

    if np.unique(time_bins).shape[0] > 1:
        raise ValueError('pc_epoch_grid should have uniform binning.')

    else:
        time_bin = np.unique(time_bins)[0]

    mjd_cent = obs_mjd[list(obs_flux).index(max(obs_flux))]
    epochs = obs_mjd - mjd_cent

    for i in range(pc_epoch_grid.shape[0]):
        flag1 = epochs >= pc_epoch_grid[i] - 0.5 * time_bin
        flag2 = epochs < pc_epoch_grid[i] + 0.5 * time_bin
        flag3 = np.logical_and(flag1, flag2)

        if sum(flag3) > 0:
            flux_final.append(np.mean(obs_flux[flag3]))
            mjd_flag.append(True)
        else:
            mjd_flag.append(False)

    if sum(mjd_flag) > 0:
        mjd_flag = np.array(mjd_flag)

        new_mjd = pc_epoch_grid[mjd_flag]
        new_flux = np.array(flux_final)

        return new_mjd, new_flux, mjd_flag, mjd_cent

    else:
        return [], [], None, None


def extract_features(
        mjd: np.array, flux: np.array, epoch_lim: list,
        time_bin: float, pcs: pd.DataFrame,
        flux_lim=0):
    """
    Extract features from light curve.

    Parameters
    ----------
    mjd: np.array
        Values for MJD.
    flux: np.array
        Values for FLUXCAL.
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim].
    time_bin: float
        Width of time gap between two elements in PCs.
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...),
        values their amplitude at each epoch in the grid.
    flux_lim: float (optional)
        Min flux cut applied to all points. Default is 0.

    Returns
    -------
    features: np.array
        Features for this light curve. Order is:
        [n_points, residual_from_fit, coefficients, max_flux]
    """

    # create list for storing output
    features = []
    mjd0 = None

    # get useful flux
    flux_flag = flux >= flux_lim

    # construct epoch grid
    pc_epoch_grid = np.arange(epoch_lim[0], epoch_lim[1] + time_bin, time_bin)

    npcs = len(pcs.keys())
    if sum(flux_flag) >= npcs:

        # translate point to suitable grid
        new_mjd, new_flux, mjd_flag, mjd0 = \
            filter_points(
                obs_mjd=mjd, obs_flux=flux,
                pc_epoch_grid=pc_epoch_grid)

        coef_mat = pd.DataFrame()
        for key in pcs.keys():
            coef_mat[key] = pcs[key].values[mjd_flag]

        # fit coefficients
        max_newflux = max(new_flux)

        x, res, rank, s = np.linalg.lstsq(
            coef_mat.values,
            new_flux / max_newflux,
            rcond=None
        )

        # add number of points and residuals and
        # coefficients to the matrix
        features.append(len(new_mjd))

        if len(res) > 0:
            features.append(res[0])
        else:
            features.append(0)

        for elem in x:
            features.append(elem)

        features.append(max_newflux)

    else:
        features = [0 for i in range(npcs + 3)]

    return features


def extract_all_filters_fink(
        epoch_lim: list, pcs: pd.DataFrame,
        time_bin: float, filters: list,
        lc: pd.DataFrame, flux_lim=0):
    """Extract features from 1 object in all available filters.

    Parameters
    ----------
    epoch_lim: list
        Min and max epoch since maximum brightness to consider.
        Format is [lower_lim, upper_lim].
    filters: list
        List of broad band filters.
    lc: pd.DataFrame
        Keys should be ['MJD', 'FLUXCAL', 'FLT'].
    pcs: pd.DataFrame
        All principal components to be considered.
        keys should be PCs names (1, 2, 3, ...),
        values their amplitude at each epoch in the grid.
        Order of PCs when calling pcs.keys() is important.
    time_bin: float
        Width of time gap between two elements in PCs.
    flux_lim: float (optional)
        Min flux cut applied to all points. Default is 0.

    Returns
    -------
    all_features: list
        List of features for this object.
        Order is all features from first filter, then all features from
        second filters, etc.
    """
    # store results from extract_features
    all_features = []

    for i in range(len(filters)):
        filter_flag = lc['FLT'].values == filters[i]

        obs_mjd = lc['MJD'].values[filter_flag]
        obs_flux = lc['FLUXCAL'].values[filter_flag]

        # extract features
        res = extract_features(
            mjd=obs_mjd, flux=obs_flux,
            epoch_lim=epoch_lim,
            time_bin=time_bin, pcs=pcs,
            flux_lim=flux_lim
        )

        all_features = all_features + res

    return all_features
