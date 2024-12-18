# import numpy as np
import pandas as pd

# from fink_utils.photometry.conversion import dc_mag, apparent_flux
# from fink_utils.photometry.utils import is_source_behind
# from fink_utils.spark.utils import concat_col
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, StringType, MapType
from fink_science.standardized_flux.utils import standardized_flux_

RELEASE = 22
CTAO_PATH = 'CTAO_blazars_ztfdr{}.parquet'.format(RELEASE)

@pandas_udf(MapType(StringType(), ArrayType(DoubleType())))
def standardized_flux(candid: pd.Series, 
                      objectId: pd.Series, 
                      cdistnr: pd.Series, 
                      cmagpsf: pd.Series, 
                      csigmapsf: pd.Series, 
                      cmagnr: pd.Series, 
                      csigmagnr: pd.Series, 
                      cisdiffpos: pd.Series, 
                      cfid: pd.Series,
                      cjd: pd.Series) -> pd.Series:
    """doc"""

    CTAO_blazar = pd.read_parquet(CTAO_PATH)
    
    pdf = pd.DataFrame(
        {
            "candid": candid, 
            "objectId": objectId, 
            "cdistnr": cdistnr, 
            "cmagpsf": cmagpsf, 
            "csigmapsf": csigmapsf, 
            "cmagnr": cmagnr, 
            "csigmagnr": csigmagnr, 
            "cisdiffpos": cisdiffpos, 
            "cfid": cfid,
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
                "cdistnr": tmp["cdistnr"].to_numpy()[0],
                "cmagpsf": tmp["cmagpsf"].to_numpy()[0],
                "csigmapsf": tmp["csigmapsf"].to_numpy()[0],
                "cmagnr": tmp["cmagnr"].to_numpy()[0],
                "csigmagnr": tmp["csigmagnr"].to_numpy()[0],
                "cisdiffpos": tmp["cisdiffpos"].to_numpy()[0],
                "cfid": tmp["cfid"].to_numpy()[0],
                "cjd": tmp["cjd"].to_numpy()[0],
            }
        )
        std_flux = standardized_flux_(sub, CTAO_blazar)
        out.append({'flux': std_flux[0], 'sigma': std_flux[1]})
    
    return pd.Series(out)
