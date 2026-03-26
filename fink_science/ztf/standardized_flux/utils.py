import numpy as np
import pandas as pd

from fink_utils.photometry.conversion import apparent_flux


def standardized_flux_(pdf: pd.DataFrame, CTAO_blazar: pd.DataFrame) -> tuple:
    """Returns the standardized flux and its uncertainties for a batch of alerts

    Parameters
    ----------
    pdf: pd.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf,
        cmagnr, csigmagnr, cisdiffpos, cfid, cjd
    CTAO_blazar : pd.DataFrame
        Pandas DataFrame of the monitored sources containing:
        ``Source_name``, ``ZTF_name``, ``medians``,
        ``low_threshold``, ``high_threshold``.

    Returns
    -------
    Tuple of pandas.Series
        Standardized flux and its uncertainties

    Notes
    -----
    Standardized flux means flux over median of each band.
    """
    std_flux = np.full(len(pdf), np.nan)
    sigma_std_flux = np.full(len(pdf), np.nan)

    name = pdf["objectId"].to_numpy()[0]
    CTAO_data = CTAO_blazar.loc[CTAO_blazar["ZTF_name"] == name]
    if not CTAO_data.empty:
        flux_dc, sigma_flux_dc = np.transpose([
            apparent_flux(*args)
            for args in zip(
                pdf["cmagpsf"].astype(float).to_numpy(),
                pdf["csigmapsf"].astype(float).to_numpy(),
                pdf["cmagnr"].astype(float).to_numpy(),
                pdf["csigmagnr"].astype(float).to_numpy(),
                pdf["cisdiffpos"].to_numpy(),
            )
        ])

        # Loop over g & r only
        for filter_ in [1, 2]:
            maskFilt = pdf["cfid"] == filter_
            median = CTAO_data["medians"].iloc[0][str(filter_)]
            std_flux[maskFilt] = flux_dc[maskFilt] / median
            sigma_std_flux[maskFilt] = sigma_flux_dc[maskFilt] / median
        print(
            name, ":", np.min(std_flux), "-", np.median(std_flux), "-", np.max(std_flux)
        )
        return pd.Series(std_flux), pd.Series(sigma_std_flux)

    else:
        return np.array([]), np.array([])
