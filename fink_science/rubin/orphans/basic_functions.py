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
import pandas as pd


def flux_to_mag(flux):
    """
    Convert flux from milliJansky to AB Magnitude

    1 Jy = 1e-23 erg/cm2/s/Hz
    Fnu = 3631 Jy = 3.631*1e-20 erg/cm2/s/Hz
    ABmag = 0-2.5*log10( Fnu )-48.6 = 0

    Parameters
    ----------
    flux: float
        Flux in milli-Jansky

    Returns
    -------
    mag: float
        Corresponding AB Magnitude
    """
    mag = -2.5 * np.log10(flux * 1.0e-26) - 48.6
    return mag


def mag_to_flux(mag):
    """
    Convert AB Magnitude to flux in milliJansky

    1 Jy = 1e-23 erg/cm2/s/Hz
    Fnu = 3631 Jy = 3.631*1e-20 erg/cm2/s/Hz
    ABmag = 0-2.5*log10( Fnu )-48.6 = 0

    Parameters
    ----------
    mag: float
        AB Magnitude

    Returns
    -------
    flux: float
        Corresponding flux in milliJansky
    """
    flux = pow(10, (26 - (mag + 48.6) / 2.5))
    return flux


def clean_and_sort_light_curve(times, mags, mags_err, filts):
    """
    Sort the points by time and remove NaN points

    Parameters
    ----------
    time: pandas.Series of list of float
        Concatenated times in MJD for the object
    mags: pandas.Series of list of float
        Concatenated magnitudes for the object
    mags_err: pandas.Series of list of float
        Concatenated magnitude errors for the object
    filt: pandas.Series of list of float
        Concatenated filters for the object

    Returns
    -------
    cleaned_times: pandas.Series of list of float
        Concatenated times in MJD for the object, sorted and without NaN points
    cleaned_mags: pandas.Series of list of float
        Concatenated magnitudes for the object, sorted and without NaN points
    cleaned_errors: pandas.Series of list of float
        Concatenated magnitude errors for the object, sorted and without NaN points
    cleaned_filts: pandas.Series of list of float
        Concatenated filters for the object, sorted and without NaN points
    """
    cleaned_times = []
    cleaned_magnitudes = []
    cleaned_errors = []
    cleaned_filters = []

    for t, m, e, f in zip(times, mags, mags_err, filts):
        # Convert to numpy arrays
        t = np.array(t)
        m = np.array(m)
        e = np.array(e)
        f = np.array(f)

        # Create a mask for non-NaN magnitudes
        mask = ~np.isnan(m)

        # Apply the mask to both times and magnitudes
        valid_times = t[mask]
        valid_mags = m[mask]
        valid_errs = e[mask]
        valid_filts = f[mask]

        # if len(valid_mags) > 4:
        # Sort the cleaned data by time
        sorted_indices = np.argsort(valid_times)
        sorted_times = valid_times[sorted_indices]
        sorted_magnitudes = valid_mags[sorted_indices]
        sorted_errors = valid_errs[sorted_indices]
        sorted_filters = valid_filts[sorted_indices]

        cleaned_times.append(sorted_times)
        cleaned_magnitudes.append(sorted_magnitudes)
        cleaned_errors.append(sorted_errors)
        cleaned_filters.append(sorted_filters)

    return (
        pd.Series(cleaned_times),
        pd.Series(cleaned_magnitudes),
        pd.Series(cleaned_errors),
        pd.Series(cleaned_filters),
    )
