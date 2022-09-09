from fink_science.agn_elasticc.classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests


@pandas_udf(DoubleType())
def agn_spark(diaObjectId, cmidPoinTai, cpsFlux, cpsFluxErr, cfilterName, ra, decl, hostgal_zphot, hostgal_zphot_err, hostgal_ra, hostgal_dec):
    
    """High level spark wrapper for the AGN classifier on ELASTiCC data

    Parameters
    ----------

    diaObjectId: Spark DataFrame Column
        Identification numbers of the objects
    cmidPoinTai: Spark DataFrame Column
        JD times (vectors of floats)
    cpsFlux, cpsFluxErr: Spark DataFrame Columns
        Flux and flux error from photometry (vectors of floats)
    cfilterName: Spark DataFrame Column
        Filter IDs (vectors of ints)
    ra: Spark DataFrame Column
        Right ascension of the objects
    decl: Spark DataFrame Column
        Declination of the objects
    hostgal_zphot, hostgal_zphot_err: Spark DataFrame Column
        Redshift and redshift error of the host galaxy
        -9 if object is in the milky way
    hostgal_ra: Spark DataFrame Column
        Right ascension of the host galaxy
        -999 if object is in the milky way
    hostgal_dec: Spark DataFrame Column
        Declination ascension of the host galaxy
        -999 if object is in the milky way

    Returns
    -------
    np.array
        ordered probabilities of being an AGN
        Return -1 if the minimum points number is not respected.

    """
    
    data = pd.DataFrame(
        {
            "objectId": diaObjectId,
            "cjd": cmidPoinTai,
            "cflux": cpsFlux,
            "csigflux": cpsFluxErr,
            "cfid": cfilterName,
            "ra": ra,
            "dec": decl,
            "hostgal_zphot": hostgal_zphot, 
            "hostgal_zphot_err": hostgal_zphot_err,
            "hostgal_ra": hostgal_ra,
            "hostgal_dec": hostgal_dec
        }
    )

    proba = agn_classifier(data)
    return pd.Series(proba)


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/agn_elasticc_alerts.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)