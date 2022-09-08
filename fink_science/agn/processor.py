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
# limitations under the License

from fink_science.agn.classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import os
from fink_science import __file__
from fink_science.tester import spark_unit_tests


@pandas_udf(DoubleType())
def agn_spark(objectId, jd, magpsf, sigmapsf, fid, ra, dec):

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
        Return -1 if the minimum number of point per passband
        (specified in kernel.py) is not respected.


    Examples
    --------
    >>> df = spark.read.format('parquet').load(ztf_alert_sample)
    >>> df_agn = df.withColumn('proba', agn_spark(df.objectId,\
                                                df.cjd,\
                                                df.cmagpsf,\
                                                df.csigmapsf,\
                                                df.cfid,\
                                                df.ra, df.dec))

    >>> df_agn.filter(df_agn['proba'] != -1).count()
    1
    >>> df_agn.filter(df_agn['proba'] == -1.0).count()
    1
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

    proba = agn_classifier(data)
    return pd.Series(proba)


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/agn_example.parquet".format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
