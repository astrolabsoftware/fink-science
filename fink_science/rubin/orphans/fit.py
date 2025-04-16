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

import numpy as np

from iminuit import Minuit
from iminuit.cost import LeastSquares

from fink_science.rubin.orphans.basic_functions import flux_to_mag, mag_to_flux


def rescale_filters(times, mags, mags_err, filts):
    """
    Calculate the rescaled magnitude to the r-band

    Parameters
    ----------
    times: list of float
        Concatenated times in MJD for the object
    mags: list of float
        Concatenated magnitudes for the object
    mags_err: list of float
        Concatenated magnitude errors for the object
    filts: list of float
        Concatenated filters for the object

    Returns
    -------
    time: array
        Array containing all the times of the rescaled points from 0 (0 being the time of the first alert)
    mag_r: array
        Array containing the rescaled magnitudes
    err: array
        Array containing the errors associated to the points
    """
    # colors and mean frequency of the band u, g, r, i, z, y
    filters = ["u", "g", "r", "i", "z", "Y"]
    all_mean_nu = [
        840336134453781.4,
        629326620516047.8,
        482703137570394.2,
        397614314115308.1,
        344530577088716.56,
        298760145396604.1,
    ]

    filt_obs = filts[filts != "r"]

    if len(filt_obs) != 0:
        unique, counts = np.unique(filt_obs, return_counts=True)
        filt_max = unique[np.argmax(counts)]
        nu_filtmax = all_mean_nu[filters.index(filt_max)]

        mag_r = mags[filts != "r"]
        time_r = times[filts != "r"]

        mag_filtmax = mags[filts == filt_max]
        time_filtmax = times[filts == filt_max]

        flux_r = mag_to_flux(mag_r)
        flux_filtmax = mag_to_flux(mag_filtmax)

        # choose values of -beta between -(p-1)/2 and -p/2, p being the electron energy population index
        p = 2.2  # p is between 2. and 2.5, we choose a standard value here
        beta = np.linspace(-(p - 1) / 2, -p / 2, 10)

        d = []

        for b in beta:
            # compute the rescaled flux for each beta
            flux_rescaled = flux_filtmax * (all_mean_nu[2] / nu_filtmax) ** (b)

            # compute the euclidean distance between the rescaled flux and the true r-band flux
            d.append(
                np.sum(
                    np.sqrt(
                        (time_filtmax[:, np.newaxis] - time_r[np.newaxis, :]) ** 2
                        + (flux_rescaled[:, np.newaxis] - flux_r[np.newaxis, :]) ** 2
                    )
                )
            )

        beta_min = beta[np.where(d == min(d))]

        all_mag_r = []
        all_err = []
        all_time = []

        # sort the magnitudes, flux, times and errors based on their filter
        for f, nu in zip(filters, all_mean_nu):
            mag_f = mags[filts == f]
            err_f = mags_err[filts == f]
            time_f = times[filts == f]

            flux_f = mag_to_flux(mag_f)

            # for the beta that minimize the distance, rescale the flux for all the bands
            if f != "r":
                all_mag_r.append(
                    flux_to_mag(flux_f * (all_mean_nu[2] / nu) ** min(beta_min))
                )
            else:
                all_mag_r.append(flux_to_mag(flux_f))

            all_time.append(time_f)
            all_err.append(err_f)

        # create one array with all the times, one with all the mag in r-band and the associated error
        time = np.concatenate(all_time)
        mag_r = np.concatenate(all_mag_r)
        err = np.concatenate(all_err)

        return time - min(time), mag_r, err

    else:
        return times - min(times), mags, mags_err


def model(t, params):
    """Function used to fit the light curve

    Parameters
    ----------
    t: list of float
        Concatenated time for the object
    params: tuple of float
        Parameters of the function, to estimate
    """
    A, B, C, D = params
    return A * t + B + C * np.exp(-D * t)


def fit_light_curve(times, mags, mags_err, filts):
    """
    Fit the light curve using an MCMC method

    Parameters
    ----------
    time: list of float
        Concatenated times in MJD for the object
    mags: list of float
        Concatenated magnitudes for the object
    err: list of float
        Concatenated magnitude errors for the object
    filt: list of float
        Concatenated filters for the object

    Returns
    -------
    A, B, C, D: float
        Parameters of the function to fit
    chi2: float
        Chi square calculated as chi2 = sum(((y - y_fit) / y_err)²)
    """
    t, y, yerr = rescale_filters(times, mags, mags_err, filts)

    least_squares = LeastSquares(t, y, yerr, model)

    initial = np.array([0.02, 20.0, 0.0, 1.5])
    m = Minuit(least_squares, initial, name=("A", "B", "C", "D"))

    m.limits = [(-1e4, 1e4), (-1e6, 1e6), (-1e7, 1e7), (-0.03, 10.0)]

    m.migrad()  # finds minimum of least_squares function
    m.hesse()  # accurately computes uncertainties

    n_try = 0
    while m.fmin.reduced_chi2 > 3.0:
        if n_try < 10:
            n_try += 1
            initial += 0.5
            m = Minuit(least_squares, initial, name=("A", "B", "C", "D"))

            m.limits = [(-1e4, 1e4), (-1e6, 1e6), (-1e7, 1e7), (-0.03, 10.0)]

            m.migrad()  # finds minimum of least_squares function
            m.hesse()  # accurately computes uncertainties
        else:
            break

    A = m.values[0]
    B = m.values[1]
    C = m.values[2]
    D = m.values[3]
    chi2 = m.fmin.reduced_chi2

    return A, B, C, D, chi2
