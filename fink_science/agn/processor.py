from classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd


@pandas_udf(DoubleType())
def agn_spark(objectId, jd, magpsf, sigmapsf, fid):
    
    data = pd.DataFrame({
        'objectId': objectId,
        'cjd':jd,
        'cmagpsf':magpsf,
        'csigmapsf':sigmapsf,
        'cfid':fid
    })
    
    proba = agn_classifier(data)
    return pd.Series(proba)