# Copyright 2020-2023 AstroLab Software
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
from fink_science import __file__
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType
from fink_science.tester import spark_unit_tests
import numpy as np
import pandas as pd
import fink_science.ztf.superluminous.slsn_classifier as slsn
from fink_science.ztf.superluminous.kernel import classifier_path
import joblib
import os


@pandas_udf(DoubleType())
@profile
def superluminous_score(
    cjd: pd.Series,
    cfid: pd.Series,
    cmagpsf: pd.Series,
    csigmapsf: pd.Series,
    candidate: pd.Series,
    is_transient: pd.Series,
) -> pd.Series:
    """High level spark wrapper for the superluminous classifier on ztf data

    Parameters
    ----------
    cjd: Spark DataFrame Column
        JD times (vectors of floats)
    cfid: Spark DataFrame Column
        Filter IDs (vectors of str)
    cmagpsf, csigmapsf: Spark DataFrame Columns
        Magnitude and magnitude error from photometry (vectors of floats)
    candidate: Spark DataFrame Column
        Additionnal info about the source. Contains distnr, the angular
        distance to the nearest reference source.
    is_transient: Spark DataFrame Column
        Is the source likely a transient.

    Returns
    -------
    np.array
        Superluminous supernovae classification probability vector
        Return 0 if not enough points were available for feature extraction

    Examples
    --------
    >>> from fink_utils.spark.utils import concat_col
    >>> from pyspark.sql import functions as F
    >>> from fink_filters.ztf.filter_transient_complete.filter import transient_complete_filter
    >>> from fink_science.ztf.transient_features.processor import extract_transient_features
    >>> sdf = spark.read.load(ztf_alert_sample)
    >>> sdf = extract_transient_features(sdf)
    >>> sdf = sdf.withColumn(
    ... "is_transient",
    ... transient_complete_filter(
    ... "faint", "positivesubtraction", "real", "pointunderneath",
    ... "brightstar", "variablesource", "stationary", "roid"))

    # Required alert columns
    >>> what = ['jd', 'fid', 'magpsf', 'sigmapsf']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...     sdf = concat_col(sdf, colname, prefix=prefix)

    # Perform the fit + classification (default model)
    >>> args = [F.col(i) for i in what_prefix]
    >>> args += ["candidate", "is_transient"]
    >>> sdf = sdf.withColumn('proba', superluminous_score(*args))
    >>> sdf.filter(sdf['proba']==-1).count()
    55
    >>> sdf.filter(sdf['proba']==0).count()
    2
    """
    pdf = pd.DataFrame(
        {
            "cjd": cjd,
            "cmagpsf": cmagpsf,
            "csigmapsf": csigmapsf,
            "cfid": cfid,
            "distnr": candidate["distnr"],
            "is_transient": is_transient,
        }
    )

    # If no alert pass the transient filter,
    # directly return invalid value for everyone.
    if sum(pdf["is_transient"]) == 0:
        return pd.Series([-1]*len(pdf))

    else:
        # Assign default 0 proba for every alert
        probas = np.zeros(len(pdf))

        # Assign -1 for non-transient alerts
        probas[~pdf["is_transient"]] = -1

        pdf = slsn.compute_flux(pdf)
        pdf = slsn.remove_nan(pdf)

        # Perform feature extraction
        features = slsn.extract_features(pdf)

        # Load classifier
        clf = joblib.load(classifier_path)

        # Modify proba for alerts that were feature extracted
        extracted = np.sum(features.isnull(), axis=1) == 0
        probas[extracted] = clf.predict_proba(
            features.loc[extracted, clf.feature_names_in_]
        )[:, 1]

        return pd.Series(probas)


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/datatest/part-00003-bdab8e46-89c4-4ac1-8603-facd71833e8a-c000.snappy.parquet".format(
        path
    )
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
