import numpy as np
import pandas as pd

#from fink_utils.photometry.conversion import dc_mag, apparent_flux
#from fink_utils.photometry.utils import is_source_behind
#from fink_utils.spark.utils import concat_col
#from pyspark.sql.functions import pandas_udf
#from pyspark.sql.types import ArrayType, DoubleType, BooleanType, StringType, MapType


#RELEASE = 22
#CTAO_PATH = 'CTAO_blazars_ztfdr{}.parquet'.format(RELEASE)

'''
parDF=spark.read.parquet("blazar_test_dataset_202411_12_agg.parquet") #?

COLUMNS = [
    'distnr', 
    'magpsf', 
    'sigmapsf', 
    'magnr', 
    'sigmagnr', 
    'isdiffpos', 
    'fid',
    'jd'
]

for key in COLUMNS:
    parDF = concat_col(parDF, colname=key, prefix="c")

CCOLUMNS = [
    'candid',
    'objectId',
    'cdistnr',  
    'cmagpsf', 
    'csigmapsf', 
    'cmagnr', 
    'csigmagnr', 
    'cisdiffpos', 
    'cfid', 
    'cjd'
]

parDF = parDF.select(CCOLUMNS)
'''

def instantness_criterion(pdf: pd.DataFrame, CTAO_blazar: pd.DataFrame) -> np.float64:
    """Returns the standardized flux of the last measurement over the precomputed threshold ratio

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
    out: np.float64
        Ratio of the standardized flux coming from the last meassurement alert over precomputed threshold
    """

    name = pdf['objectId'].values[0] 
    
    try:
        threshold = np.array(CTAO_blazar.loc[CTAO_blazar['ZTF Name'] == name, 'Final Threshold'].values[0])
    except IndexError:
        threshold = np.nan

    try:
        return pdf['cstd_flux'].iloc[-1] / threshold 
    except KeyError:
        return np.nan


def robustness_criterion(pdf: pd.DataFrame, CTAO_blazar: pd.DataFrame) -> np.float64:
    """Returns the sliding mean over 30 days of the standardized flux over the precomputed threshold ratio

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
    out: np.float64
        Ratio of the sliding mean over 30 days of the standardized flux over the precomputed threshold
    """

    integration_period = 30
    name = pdf['objectId'].values[0]
    
    try:
        threshold = np.array(CTAO_blazar.loc[CTAO_blazar['ZTF Name'] == name, 'Final Threshold'].values[0])
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
    if maskNan.sum() > 1:
        return (np.trapz(flux[maskNan], x=time[maskNan]) 
                / (time[maskNan].iloc[-1] - time[maskNan].iloc[0]) 
                / threshold)
    else:
        return np.nan

def low_state_(pdf: pd.DataFrame, CTAO_blazar: pd.DataFrame) -> np.ndarray:
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
    out: np.ndarray[np.float64]
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
