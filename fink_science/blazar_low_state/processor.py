from line_profiler import profile

import pandas as pd

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType
from fink_science.blazar_low_state.utils import quiescent_state_


RELEASE = 22
CTAO_PATH = 'CTAO_blazars_ztf_dr{}.parquet'.format(RELEASE)


@pandas_udf(ArrayType(DoubleType()))
@profile
def quiescent_state(candid: pd.Series, 
              objectId: pd.Series, 
              cstd_flux: pd.Series, 
              cjd: pd.Series) -> pd.Series:
    """Returns an array containing:
            The mean over threshold ratio of the last but one alert
            The mean over threshold ratio of the last alert
            The standardized flux over threshold ratio of the last alert

    Parameters
    ----------
    pdf: pd.core.frame.DataFrame
        Pandas DataFrame of the alert history containing:
        candid, ojbectId, cdistnr, cmagpsf, csigmapsf, cmagnr, csigmagnr, cisdiffpos, cfid, cjd, cstd_flux, csigma_std_flux
    CTAO_blazar: pd.core.frame.DataFrame
        Pandas DataFrame of the monitored sources containing:
        3FGL Name, ZTF Name, Arrays of Medians, Computed Threshold, Observed Threshold, Redshift, Final Threshold
    Returns
    -------
    out: pd.Series of np.ndarray of np.float64
        Array of ratios for:
        Mean over threshold of the last but one alert
        Mean over threshold of the last alert
        Measurement over threshold of the last alert
    """

    CTAO_blazar = pd.read_parquet(CTAO_PATH)
    pdf = pd.DataFrame(
        {
            "candid": candid, 
            "objectId": objectId, 
            "cstd_flux": cstd_flux, 
            "cjd": cjd
        }
    )
    out = []
    for candid_ in pdf["candid"]:
        tmp = pdf[pdf["candid"] == candid_]
        sub = pd.DataFrame(
            {
                "candid": tmp["candid"].to_numpy()[0],
                "objectId": tmp["objectId"].to_numpy()[0],
                "cstd_flux": tmp["cstd_flux"].to_numpy()[0],
                "cjd": tmp["cjd"].to_numpy()[0],
            }
        )
        out.append(quiescent_state_(sub, CTAO_blazar))
    
    return pd.Series(out)
