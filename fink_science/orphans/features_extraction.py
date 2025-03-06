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


def compute_duration_between_first_and_peak(times, mags):
    """
    Save the number of days between the first detection and the minimal detected magnitude

    Parameters
    ----------
    times: list of float
        Concatenated times in MJD for the object
    mags: list of float
        Concatenated magnitudes for the object

    Returns
    -------
    Dt: list
        List containing the number of days between the first detection and the peak of each configuration in each filter
    """

    # calculate the duration between the first detection and the peak
    if (len(mags) != 0) and (np.min(mags) != mags[0]):
        Dt = (times[np.where(mags == np.min(mags))] - min(times))[0]

    else:
        Dt = 0.

    return Dt


def compute_rates(times, mags, filts):
    """
    Save the increase rate (in mag/day) + the decrease rates (in mag/day) in the first third and the last third of
    the decreasing part of the light curve

    Parameters
    ----------
    times: list of float
        Concatenated times in MJD for the object
    mags: list of float
        Concatenated magnitudes for the object
    filts: list of float
        Concatenated filters for the object

    Returns
    -------
    rate[0]: float
        Increase rate of the light curve
    rate[1]: float
        Decrease rate of the light curve
    rate[2]: float
        Decrease rate in the 1/3 of the decreasing part of the light curve
    rate[3]: float
        Decrease rate in the 3/3 of the decreasing part of the light curve
    """

    filters = ['u', 'g', 'r', 'i', 'z', 'Y']

    increase_rate = np.array([])
    first_third_decrease_rate = np.array([])
    last_third_decrease_rate = np.array([])

    # calculate rates for each filter
    for f in filters:

        time_f = times[np.where(filts == f)[0]]
        mag_f = mags[np.where(filts == f)[0]]

        delta_time = np.diff(time_f)
        delta_mag = np.diff(mag_f)

        rate_of_change = delta_mag[delta_time > 0.1] / delta_time[delta_time > 0.1]

        # Separating increase and decrease rates
        if rate_of_change[rate_of_change < 0].size != 0:
            increase_rate = np.append(increase_rate, rate_of_change[rate_of_change < 0])
        else:
            increase_rate = np.append(increase_rate, 0.)

        # indices where the rate is decreasing
        decreasing_indices = np.where(rate_of_change > 0)[0]

        # split the decreasing part into thirds
        n_decreasing = decreasing_indices.size
        if n_decreasing >= 3:
            first_third_indices = decreasing_indices[:n_decreasing // 3]
            last_third_indices = decreasing_indices[-(n_decreasing // 3):]

            # calculate the decrease rates for the first and last thirds
            first_third_decrease_rate = np.append(first_third_decrease_rate, rate_of_change[first_third_indices])
            last_third_decrease_rate = np.append(last_third_decrease_rate, rate_of_change[last_third_indices])
        else:
            first_third_decrease_rate = np.append(first_third_decrease_rate, 0.)
            last_third_decrease_rate = np.append(last_third_decrease_rate, 0.)

    rate = []

    # compute the mean rates
    for r in [increase_rate, first_third_decrease_rate, last_third_decrease_rate]:
        if np.any(r) != 0.:
            np.array(r)[np.array(r) == 0] = np.nan
            rate.append(np.nanmean(r))
        else:
            rate.append(0.)

    return rate[0], rate[1], rate[2]


def compute_colours(times, mags, filts):
    """
    Calculate the colour for 2 pairs of filters: g-r and r-i.

    Parameters
    ----------
    times: list of float
        Concatenated times in MJD for the object
    mags: list of float
        Concatenated magnitudes for the object
    filts: list of float
        Concatenated filters for the object

    Returns
    -------
    mean_colours: array
        Array containing the mean colours for the filter pairs g-r and r-i
    """

    mean_colours = np.array([])

    filter_pairs = [('g', 'r'), ('r', 'i')]

    for pair in filter_pairs:
        filter1, filter2 = pair

        # get magnitudes and times for each filter in the pair
        time_filter1 = times[np.where(filts == filter1)[0]]
        mag_filter1 = mags[np.where(filts == filter1)[0]]

        time_filter2 = times[np.where(filts == filter2)[0]]
        mag_filter2 = mags[np.where(filts == filter2)[0]]

        if mag_filter1.size == 0 or mag_filter2.size == 0:
            # if data for either filter is not available, set color index to NaN
            mean_colours = np.append(mean_colours, np.nan)
            continue

        # interpolate magnitudes for filter1 at times of filter2
        interpolated_mag_filter1 = np.interp(time_filter2, time_filter1, mag_filter1)

        # calculate color index as the difference between magnitudes
        colour_index = interpolated_mag_filter1 - mag_filter2

        # store the mean color index for the current filter pair
        mean_colours = np.append(colour, np.mean(colour_index))

    return mean_colours