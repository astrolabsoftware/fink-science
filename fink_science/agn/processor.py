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
<<<<<<< HEAD

from fink_science.agn.classifier import agn_classifier
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import os
from fink_science import __file__
=======
import os
import pandas as pd
import numpy as np

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

from fink_science import __file__
from fink_science.agn.classifier import agn_classifier
>>>>>>> f0784aba8b520c42198cf869249ccc4540e20826
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
<<<<<<< HEAD
        Return -1 if the minimum number of point per passband
=======
        Return 0 if the minimum number of point per passband
>>>>>>> f0784aba8b520c42198cf869249ccc4540e20826
        (specified in kernel.py) is not respected.


    Examples
    --------
<<<<<<< HEAD
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
=======
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    # Required alert columns
    >>> what = ['jd', 'magpsf', 'sigmapsf', 'fid']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = ['objectId'] + [F.col(i) for i in what_prefix]
    >>> args += ['candidate.ra', 'candidate.dec']
    >>> df_agn = df.withColumn('proba', agn_spark(*args))

    >>> df_agn.filter(df_agn['proba'] != 0.0).count()
    145

    >>> df_agn.filter(df_agn['proba'] == 0.0).count()
    175

    >>> df_agn.filter(df_agn['proba'] > 0.5).count()
    26
    """
    # we want at least 2 bands and 4 points per band
    nbands = fid.apply(lambda x: len(np.unique(x)))

    ng = fid.apply(lambda x: np.sum(np.array(x) == 1))
    nr = fid.apply(lambda x: np.sum(np.array(x) == 2))

    mask = (nbands == 2) & (ng >= 4) & (nr >= 4)

    if len(objectId[mask]) == 0:
        return pd.Series(np.zeros(len(objectId), dtype=float))
>>>>>>> f0784aba8b520c42198cf869249ccc4540e20826

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

<<<<<<< HEAD
    proba = agn_classifier(data)
    return pd.Series(proba)
=======
    proba = agn_classifier(data[mask])

    to_return = np.zeros(len(jd), dtype=float)
    to_return[mask] = proba
    return pd.Series(to_return)
>>>>>>> f0784aba8b520c42198cf869249ccc4540e20826


if __name__ == "__main__":

    globs = globals()
    path = os.path.dirname(__file__)

<<<<<<< HEAD
    ztf_alert_sample = "file://{}/data/alerts/agn_example.parquet".format(path)
=======
    ztf_alert_sample = 'file://{}/data/alerts/datatest'.format(path)
>>>>>>> f0784aba8b520c42198cf869249ccc4540e20826
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
