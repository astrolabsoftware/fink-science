# Copyright 2022 Fink Software
# Author: Etienne Russeil
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fink_science.agn.classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests


@pandas_udf(DoubleType())
def agn_elasticc(
        diaObjectId, cmidPoinTai, cpsFlux, cpsFluxErr, cfilterName,
        ra, decl, hostgal_zphot, hostgal_zphot_err, hostgal_ra, hostgal_dec):
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
    model_path: Spark DataFrame Column, optional
        Path to the model. If None (default), it is
        taken from `k.CLASSIFIER`.

    Returns
    -------
    np.array
        ordered probabilities of being an AGN
        Return 0 if the minimum points number is not respected.
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

    proba = agn_classifier(data, source='ELASTICC')
    return pd.Series(proba)


@pandas_udf(DoubleType())
def agn_ztf(objectId, jd, magpsf, sigmapsf, fid, ra, dec):
    """High level spark wrapper for the AGN classifier on ZTF data

    Parameters
    ----------
    objectId: Spark DataFrame Column
        Identification numbers of the objects
    jd: Spark DataFrame Column
        JD times (vectors of floats)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry,
        and 1-sigma error (vectors of floats)
    fid: Spark DataFrame Column
        Filter IDs (vectors of ints)
    ra: Spark DataFrame Column
        Right ascension of the objects
    dec: Spark DataFrame Column
        Declination of the objects

    Returns
    -------
    np.array
        ordered probabilities of being an AGN
        Return 0 if the minimum number of point per passband
        (specified in kernel.py) if not respected.

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    # Required alert columns
    >>> what = ['jd', 'magpsf', 'sigmapsf', 'fid']

    >>> df = spark.read.load(ztf_alert_sample)

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = ['objectId'] + [F.col(i) for i in what_prefix]
    >>> args += ['candidate.ra', 'candidate.dec']
    >>> df_agn = df.withColumn('proba', agn_ztf(*args))
    >>> df_agn.filter(df_agn['proba'] != 0.0).count()
    145
    """

    data = pd.DataFrame(
        {
            "objectId": objectId,
            "cjd": jd,
            "cmagpsf": magpsf,
            "csigmapsf": sigmapsf,
            "cfid": fid,
            "ra": ra,
            "dec": dec,
        }
    )

    proba = agn_classifier(data, source='ZTF')

    return pd.Series(proba)


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
