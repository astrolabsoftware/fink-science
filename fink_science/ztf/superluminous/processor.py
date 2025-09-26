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
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from fink_science.tester import spark_unit_tests
import numpy as np
import pandas as pd
import fink_science.ztf.superluminous.slsn_classifier as slsn
from fink_science.ztf.superluminous.kernel import classifier_path
import joblib
import os
import requests
import io


@pandas_udf(DoubleType())
@profile
def superluminous_score(is_transient: pd.Series, objectId: pd.Series) -> pd.Series:
    """High level spark wrapper for the superluminous classifier on ztf data

    Parameters
    ----------
    cjd: Spark DataFrame Column
        JD times (vectors of floats)
    cfid: Spark DataFrame Column
        Filter IDs (vectors of str)
    cmagpsf, csigmapsf: Spark DataFrame Columns
        Magnitude and magnitude error from photometry (vectors of floats)
    distnr: Spark DataFrame Column
        The angular distance to the nearest reference source.
    is_transient: Spark DataFrame Column
        Is the source likely a transient.
    objectId: Spark DataFrame Column
        Unique source ZTF name

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

    # Perform the fit + classification (default model)
    >>> args = ["is_transient", 'objectId']
    >>> sdf = sdf.withColumn('proba', superluminous_score(*args))
    >>> sdf.filter(sdf['proba']==-1.0).count()
    57
    """

    pdf = pd.DataFrame(
        {
            "is_transient": is_transient,
            "objectId": objectId,
        }
    )

  

    # If no alert pass the transient filter,
    # directly return invalid value for everyone.
    if sum(pdf["is_transient"]) == 0:
        return pd.Series([-1.0] * len(pdf))

    else:
        # Initialise all probas to -1
        probas_total = np.zeros(len(pdf), dtype=float) - 1
        mask_valid = pdf["is_transient"]

        # select only transient alerts
        pdf_valid = pdf[mask_valid]
        valid_ids = list(pdf_valid["objectId"])

        # Use Fink API to get the full light curves
        lcs = get_and_format(valid_ids)

        if lcs is not None:
            # Assign default -1 proba for every valid alert
            probas = np.zeros(len(pdf_valid), dtype=float) - 1

            lcs = slsn.compute_flux(lcs)
            lcs = slsn.remove_nan(lcs)

            # Perform feature extraction
            features = slsn.extract_features(lcs)

            # Load classifier
            clf = joblib.load(classifier_path)

            # Modify proba for alerts that were feature extracted
            extracted = np.sum(features.isnull(), axis=1) == 0
            probas[extracted] = clf.predict_proba(
                features.loc[extracted, clf.feature_names_in_]
            )[:, 1]

            probas_total[mask_valid] = probas
            return pd.Series(probas_total)

        else:
            return pd.Series([-1.0] * len(pdf))


def get_and_format(ZTF_name):
    """Use the fink API to collect the full light curve
    sources based on a list of ZTF names

    Parameters
    ----------
    ZTF_name: list
        List of objectId (ZTF names)

    Returns
    -------
    pd.DataFrame
        DataFrame containing all light curve information.
        1 row = 1 source. Returns None if the list is empty.
    """
    if len(ZTF_name) == 0:
        return None

    # Prepare the result table
    columns = ["objectId", "cjd", "cmagpsf", "csigmapsf", "cfid"]
    final_pdf = pd.DataFrame(columns=columns)

    r = requests.post(
        "https://api.fink-portal.org/api/v1/objects",
        json={
            "objectId": ",".join(ZTF_name),
            "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid,i:jd,i:distnr,d:tag",
            "output-format": "json",
            "withupperlim": "True",
        },
    )

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))

    if len(pdf) != 0:
        pdf = pdf.sort_values("i:jd")
        valid = (pdf["d:tag"] == "valid") | (pdf["d:tag"] == "badquality")
        pdf = pdf[valid]
        pdfs = [group for _, group in pdf.groupby("i:objectId")]
        lcs = pd.DataFrame(
            data={
                "objectId": ZTF_name,
                "cjd": [np.array(pdf["i:jd"].values, dtype=float) for pdf in pdfs],
                "cmagpsf": [
                    np.array(pdf["i:magpsf"].values, dtype=float) for pdf in pdfs
                ],
                "csigmapsf": [
                    np.array(pdf["i:sigmapsf"].values, dtype=float) for pdf in pdfs
                ],
                "cfid": [np.array(pdf["i:fid"].values, dtype=int) for pdf in pdfs],
                "distnr": [np.mean(pdf["i:distnr"]) for pdf in pdfs],
            }
        )
        return lcs
    return None


if __name__ == "__main__":
    globs = globals()
    path = os.path.dirname(__file__)

    ztf_alert_sample = "file://{}/data/alerts/datatest/part-00003-bdab8e46-89c4-4ac1-8603-facd71833e8a-c000.snappy.parquet".format(
        path
    )
    globs["ztf_alert_sample"] = ztf_alert_sample

    # Run the test suite
    spark_unit_tests(globs)
