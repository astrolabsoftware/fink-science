import numpy as np
import pandas as pd

from fink_utils.photometry.conversion import apparent_flux


def standardized_flux_(pdf: pd.DataFrame, CTAO_blazar: pd.DataFrame) -> tuple:
    """Returns the standardized flux and its uncertainties for a batch of alerts

    Notes
    -----
    Standardized flux means flux over median of each band.

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf,
        cmagnr, csigmagnr, cisdiffpos, cfid, cjd
    CTAO_blazar: pd.core.frame.DataFrame
        Pandas DataFrame of the monitored sources containing:
        3FGL Name, ZTF Name, Arrays of Medians, Computed Threshold,
        Observed Threshold, Redshift, Final Threshold

    Returns
    -------
    Tuple of pandas.Series
        Standardized flux and its uncertainties
    """
    std_flux = np.full(len(pdf), np.nan)
    sigma_std_flux = np.full(len(pdf), np.nan)

    name = pdf["objectId"].to_numpy()[0]
    CTAO_data = CTAO_blazar.loc[CTAO_blazar["ZTF Name"] == name]
    if not CTAO_data.empty:
        flux_dc, sigma_flux_dc = 1000 * np.transpose([
            apparent_flux(*args)
            for args in zip(
                pdf["cmagpsf"].astype(float).to_numpy(),
                pdf["csigmapsf"].astype(float).to_numpy(),
                pdf["cmagnr"].astype(float).to_numpy(),
                pdf["csigmagnr"].astype(float).to_numpy(),
                pdf["cisdiffpos"].to_numpy(),
            )
        ])

        for filter in pdf["cfid"].unique():
            maskFilt = pdf["cfid"] == filter
            median = CTAO_data["Array of Medians"].to_numpy()[0][filter - 1]
            std_flux[maskFilt] = flux_dc[maskFilt] / median
            sigma_std_flux[maskFilt] = sigma_flux_dc[maskFilt] / median
        return pd.Series(std_flux), pd.Series(sigma_std_flux)

    else:
        return np.array([]), np.array([])
