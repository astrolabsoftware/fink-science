import pandas as pd

#from fink_utils.photometry.conversion import dc_mag, apparent_flux
#from fink_utils.photometry.utils import is_source_behind
#from fink_utils.spark.utils import concat_col
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType
from fink_science.blazar_low_state.utils import low_state_


RELEASE = 22
CTAO_PATH = 'CTAO_blazars_ztf_dr{}.parquet'.format(RELEASE)

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

@pandas_udf(ArrayType(DoubleType()))
def low_state(candid: pd.Series, 
              objectId: pd.Series, 
              cstd_flux: pd.Series, 
              cjd: pd.Series) -> pd.Series:
    """doc"""

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
        out.append(low_state_(sub, CTAO_blazar))
    
    return pd.Series(out)
