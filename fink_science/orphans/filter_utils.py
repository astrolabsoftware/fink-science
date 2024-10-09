import numpy as np
import pandas as pd

from iminuit import Minuit
from iminuit.cost import LeastSquares


# TOOLS
# =======================================================================================================================


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

        if len(valid_mags) > 4:
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

    return pd.Series(cleaned_times), pd.Series(cleaned_magnitudes), pd.Series(cleaned_errors), pd.Series(
        cleaned_filters)


# LIGHT CURVE FEATURES
# =======================================================================================================================


def compute_duration_between_first_and_peak(times, mags):
    """
    Save the number of days between the first detection and the peak, and the date of the first detection

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
    Save the number of days between the first detection and the peak, and the date of the first detection

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
    decrease_rate = np.array([])
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

        if rate_of_change[rate_of_change > 0].size != 0:
            decrease_rate = np.append(decrease_rate, rate_of_change[rate_of_change > 0])
        else:
            decrease_rate = np.append(decrease_rate, 0.)

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
    for r in [increase_rate, decrease_rate, first_third_decrease_rate, last_third_decrease_rate]:
        if np.any(r) != 0.:
            np.array(r)[np.array(r) == 0] = np.nan
            rate.append(np.nanmean(r))
        else:
            rate.append(0.)

    return rate[0], rate[1], rate[2], rate[3]


def compute_colours(times, mags, filts):
    """
    Calculate the colour for pairs of filters.

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
    colour_indices: array
        Array containing the colours for the filter pairs u-g, g-r, r-i, i-z, z-y
    """

    colour = np.array([])

    filter_pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z'), ('z', 'Y')]

    for pair in filter_pairs:
        filter1, filter2 = pair

        # get magnitudes and times for each filter in the pair
        time_filter1 = times[np.where(filts == filter1)[0]]
        mag_filter1 = mags[np.where(filts == filter1)[0]]

        time_filter2 = times[np.where(filts == filter2)[0]]
        mag_filter2 = mags[np.where(filts == filter2)[0]]

        if mag_filter1.size == 0 or mag_filter2.size == 0:
            # if data for either filter is not available, set color index to NaN
            colour = np.append(colour, np.nan)
            continue

        # interpolate magnitudes for filter1 at times of filter2
        interpolated_mag_filter1 = np.interp(time_filter2, time_filter1, mag_filter1)

        # calculate color index as the difference between magnitudes
        colour_index = interpolated_mag_filter1 - mag_filter2

        # store the mean color index for the current filter pair
        colour = np.append(colour, np.mean(colour_index))

    return colour


# LIGHT CURVE FIT
# =======================================================================================================================


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
    filters = ['u', 'g', 'r', 'i', 'z', 'Y']
    all_mean_nu = [840336134453781.4, 629326620516047.8, 482703137570394.2, 397614314115308.1, 344530577088716.56,
                   298760145396604.1]

    filt_obs = filts[filts != 'r']

    if len(filt_obs) != 0:

        unique, counts = np.unique(filt_obs, return_counts=True)
        filt_max = unique[np.argmax(counts)]
        nu_filtmax = all_mean_nu[filters.index(filt_max)]

        mag_r = mags[filts != 'r']
        time_r = times[filts != 'r']

        mag_filtmax = mags[filts == filt_max]
        time_filtmax = times[filts == filt_max]

        flux_r = mag_to_flux(mag_r)
        flux_filtmax = mag_to_flux(mag_filtmax)

        # choose values of -beta between -(p-1)/2 and -p/2
        beta = np.linspace(-0.6, -1.1, 10)

        d = []

        for b in beta:
            # compute the rescaled flux for each beta
            flux_rescaled = flux_filtmax * (all_mean_nu[2] / nu_filtmax) ** (b)

            # compute the euclidean distance between the rescaled flux and the true r-band flux
            d.append(np.sum(np.sqrt((time_filtmax[:, np.newaxis] - time_r[np.newaxis, :]) ** 2 +
                                    (flux_rescaled[:, np.newaxis] - flux_r[np.newaxis, :]) ** 2)))

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
            if f != 'r':
                all_mag_r.append(flux_to_mag(flux_f * (all_mean_nu[2] / nu) ** min(beta_min)))
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

    initial = np.array([0.02, 20., 0., 1.5])
    m = Minuit(least_squares, initial, name=('A', 'B', 'C', 'D'))

    m.limits = [(-1e4, 1e4), (-1e6, 1e6), (-1e7, 1e7), (0, 3)]

    m.migrad()  # finds minimum of least_squares function
    m.hesse()  # accurately computes uncertainties

    n_try = 0
    while (m.fmin.reduced_chi2 > 3.):
        if (n_try < 10):
            n_try += 1
            initial += 0.5
            m = Minuit(least_squares, initial, name=('A', 'B', 'C', 'D'))

            m.limits = [(-1e4, 1e4), (-1e6, 1e6), (-1e7, 1e7), (0, 3)]

            m.migrad();  # finds minimum of least_squares function
            m.hesse();  # accurately computes uncertainties
        else:
            break

    A = m.values[0]
    B = m.values[1]
    C = m.values[2]
    D = m.values[3]
    chi2 = m.fmin.reduced_chi2

    return A, B, C, D, chi2