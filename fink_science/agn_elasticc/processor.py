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

from fink_science.agn_elasticc.classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests
import kernel as k


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

    passbands = ['u', 'g', 'r', 'i', 'z', 'Y']
    mask = [False] * len(diaObjectId)
    valid_filters = []

    for band in range(len(passbands)):
        valid_filters.append(cfilterName.apply(lambda x: np.sum(np.array(x) == passbands[band]) >= k.MINIMUM_POINTS))

    for band in range(len(passbands) - 1):
        mask = mask | ((valid_filters[band]) & valid_filters[band + 1])

    if len(diaObjectId[mask]) == 0:
        return pd.Series(np.zeros(len(diaObjectId), dtype=float))

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

    # agn_classifier relies on index being a range from 0 to N
    data_sub = data[mask].reset_index()

    proba = agn_classifier(data_sub, source='ELASTICC')
    to_return = np.zeros(len(cmidPoinTai), dtype=float)
    to_return[mask] = proba
    return pd.Series(to_return)


@pandas_udf(DoubleType())
def agn_ztf(objectId, jd, magpsf, sigmapsf, fid, ra, dec):

    """High level spark wrapper for the AGN classifier

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
    """

    ng = fid.apply(lambda x: np.sum(np.array(x) == 1))
    nr = fid.apply(lambda x: np.sum(np.array(x) == 2))

    mask = (ng >= 4) & (nr >= 4)

    if len(objectId[mask]) == 0:
        return pd.Series(np.zeros(len(objectId), dtype=float))

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

    proba = agn_classifier(data[mask], source='ZTF')

    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = proba
    return pd.Series(to_return)


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    # Run the test suite
    spark_unit_tests(globs)
