import numpy as np
import pandas as pd


def instantness_criterion(
        pdf: pd.DataFrame,
        CTAO_blazar: pd.DataFrame) -> np.float64:
    """Returns the standardized flux of the last measurement
       over the precomputed threshold ratio

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf, cmagnr,
        csigmagnr, cisdiffpos, cfid, cjd, cstd_flux, csigma_std_flux
    CTAO_blazar: pd.core.frame.DataFrame
        Pandas DataFrame of the monitored sources containing:
        3FGL Name, ZTF Name, Arrays of Medians, Computed Threshold,
        Observed Threshold, Redshift, Final Threshold

    Returns
    -------
    out: np.float64
        Ratio of the standardized flux coming from the last measurement alert
        over precomputed threshold
    """

    name = pdf['objectId'].values[0]

    try:
        threshold = np.array(
            CTAO_blazar.loc[
                CTAO_blazar['ZTF Name'] == name,
                'Final Threshold'
            ].values[0]
        )
    except IndexError:
        threshold = np.nan

    try:
        return pdf['cstd_flux'].iloc[-1] / threshold
    except KeyError:
        return np.nan


def robustness_criterion(
        pdf: pd.DataFrame,
        CTAO_blazar: pd.DataFrame) -> np.float64:
    """Returns the sliding mean over 30 days of the standardized flux
       over the precomputed threshold ratio

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf, cmagnr,
        csigmagnr, cisdiffpos, cfid, cjd, cstd_flux, csigma_std_flux
    CTAO_blazar: pd.core.frame.DataFrame
        Pandas DataFrame of the monitored sources containing:
        3FGL Name, ZTF Name, Arrays of Medians, Computed Threshold,
        Observed Threshold, Redshift, Final Threshold

    Returns
    -------
    out: np.float64
        Ratio of the sliding mean over 30 days of the standardized flux over
        the precomputed threshold
    """

    integration_period = 30
    name = pdf['objectId'].values[0]

    try:
        threshold = np.array(
            CTAO_blazar.loc[
                CTAO_blazar['ZTF Name'] == name,
                'Final Threshold'
            ].values[0]
        )
    except IndexError:
        threshold = np.nan

    try:
        full_time = pdf['cjd']
        maskTime = full_time >= full_time.iloc[-1] - integration_period
        time = pdf.loc[maskTime, 'cjd']
        flux = pdf.loc[maskTime, 'cstd_flux']
    except KeyError:
        return np.nan

    maskNan = ~pd.isnull(flux)
    mtime = time[maskNan]
    mtimestart = mtime.iloc[0]
    mtimestop = mtime.iloc[-1]
    if maskNan.sum() > 1:
        integral = np.trapz(flux[maskNan], x=time)
        return integral / (mtimestop - mtimestart) / threshold
    else:
        return np.nan


def quiescent_state_(
        pdf: pd.DataFrame,
        CTAO_blazar: pd.DataFrame) -> np.ndarray:
    """Returns an array containing:
            The mean over threshold ratio of the last but one alert
            The mean over threshold ratio of the last alert
            The standardized flux over threshold ratio of the last alert

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf, cmagnr,
        csigmagnr, cisdiffpos, cfid, cjd, cstd_flux, csigma_std_flux
    CTAO_blazar: pd.core.frame.DataFrame
        Pandas DataFrame of the monitored sources containing:
        3FGL Name, ZTF Name, Arrays of Medians, Computed Threshold,
        Observed Threshold, Redshift, Final Threshold
    Returns
    -------
    out: np.ndarray of np.float64
        Array of ratios for:
        Mean over threshold of the last but one alert
        Mean over threshold of the last alert
        Measurement over threshold of the last alert
    """

    name = pdf['objectId'].values[0]

    if not CTAO_blazar.loc[CTAO_blazar['ZTF Name'] == name].empty:
        return np.array(
            [
                robustness_criterion(pdf[:-1], CTAO_blazar),
                robustness_criterion(pdf, CTAO_blazar),
                instantness_criterion(pdf, CTAO_blazar)
            ]
        )

    else:
        return np.full(3, np.nan)
