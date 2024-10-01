# Copyright 2022-2024 Fink Software
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
from line_profiler import profile

from fink_science.slsn.classifier import slsn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
from fink_science.tester import spark_unit_tests


@pandas_udf(DoubleType())
@profile
def slsn_elasticc_no_md(
    diaObjectId, cmidPointTai, cpsFlux, cpsFluxErr, cfilterName, ra, decl
):
    """High level spark wrapper for the slsn classifier on ELASTiCC data

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

    Returns
    -------
    np.array
        ordered probabilities of being a slsn
        Return 0 if the minimum points number is not respected.
    """

    data = pd.DataFrame(
        {
            "diaObjectId": diaObjectId,
            "cmidPointTai": cmidPointTai,
            "cpsFlux": cpsFlux,
            "cpsFluxErr": cpsFluxErr,
            "cfilterName": cfilterName,
            "ra": ra,
            "decl": decl,
        }
    )

    proba = slsn_classifier(data, False)
    return pd.Series(proba)


@pandas_udf(DoubleType())
@profile
def slsn_elasticc_with_md(
    diaObjectId, cmidPointTai, cpsFlux, cpsFluxErr, cfilterName,
    ra, decl, hostgal_zphot, hostgal_zphot_err, hostgal_snsep
):
    """High level spark wrapper for the slsn classifier on ELASTiCC data

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
    hostgal_snsep: Spark DataFrame Column
        Distance from the host galaxy


    Returns
    -------
    np.array
        ordered probabilities of being a slsn
        Return 0 if the minimum points number is not respected.
    """

    data = pd.DataFrame(
        {
            "diaObjectId": diaObjectId,
            "cmidPointTai": cmidPointTai,
            "cpsFlux": cpsFlux,
            "cpsFluxErr": cpsFluxErr,
            "cfilterName": cfilterName,
            "ra": ra,
            "decl": decl,
            "hostgal_zphot": hostgal_zphot,
            "hostgal_zphot_err": hostgal_zphot_err,
            "hostgal_snsep": hostgal_snsep,
        }
    )

    proba = slsn_classifier(data, True)
    return pd.Series(proba)


if __name__ == "__main__":
    globs = globals()

    # Run the test suite
    spark_unit_tests(globs)
