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
def superluminous_score(
    is_transient: pd.Series,
    objectId: pd.Series,
    jdstarthist: pd.Series,
    cjd: pd.Series,
    cfid: pd.Series,
    cmagpsf: pd.Series,
    csigmapsf: pd.Series,
) -> pd.Series:
    """High level spark wrapper for the superluminous classifier on ztf data

    Parameters
    ----------
    is_transient: Spark DataFrame Column
        Is the source likely a transient.
    objectId: Spark DataFrame Column
        Unique source ZTF name
    jdstarthist: Spark DataFrame Column
        Time of first alert of this source.
    cjd: Spark DataFrame Column
        JD times (vectors of floats)
    cfid: Spark DataFrame Column
        Filter IDs (vectors of str)
    cmagpsf, csigmapsf: Spark DataFrame Columns
        Magnitude and magnitude error from photometry (vectors of floats)

    Returns
    -------
    np.array
        Superluminous supernovae classification probability vector
        Return -1 if not enough points were available for feature extraction
        if the alert is not considered a likely transient
        or if the source is less than 30 days old

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
    >>> args = ['is_transient', 'objectId', 'candidate.jdstarthist']

    # Perform the fit + classification (default model)
    >>> args += [F.col(i) for i in what_prefix]

    >>> sdf = sdf.withColumn('proba', superluminous_score(*args))
    >>> pdf = sdf.toPandas()
    >>> sum(pdf['proba']==-1)
    57
    >>> sum(pdf['is_transient'])
    2
    """
    pdf = pd.DataFrame({
        "is_transient": is_transient,
        "objectId": objectId,
        "jdstarthist": jdstarthist,
        "cjd": cjd,
        "cmagpsf": cmagpsf,
        "csigmapsf": csigmapsf,
        "cfid": cfid,
    })

    pdf["jd"] = pdf["cjd"].apply(np.max)

    # If no alert pass the transient filter,
    # directly return invalid value for everyone.
    if sum(pdf["is_transient"]) == 0:
        return pd.Series([-1.0] * len(pdf))

    else:
        # Initialise all probas to -1
        probas_total = np.zeros(len(pdf), dtype=float) - 1
        transient_mask = pdf["is_transient"]
        old_enough_mask = pdf["jd"] - pdf["jdstarthist"] >= 30
        mask_valid = transient_mask & old_enough_mask

        # select only transient alerts
        pdf_valid = pdf[mask_valid].copy().reset_index()

        valid_ids = list(pdf_valid["objectId"])

        # Use Fink API to get the full light curves history
        lcs = get_and_format(valid_ids)

        # Compute the current night alerts (anything more recent tham the last history)
        pdf_valid["last_alerts_cjd"] = lcs["cjd"].apply(lambda x: np.nanmax(x))
        pdf_valid["is_new"] = pdf_valid[["cjd", "last_alerts_cjd"]].apply(
            lambda x: x["cjd"] > x["last_alerts_cjd"], axis=1
        )

        current_night = pd.DataFrame(
            data={
                "cjd": pdf_valid[["cjd", "is_new"]].apply(
                    lambda x: x["cjd"][x["is_new"]], axis=1
                ),
                "cmagpsf": pdf_valid[["cmagpsf", "is_new"]].apply(
                    lambda x: x["cmagpsf"][x["is_new"]], axis=1
                ),
                "csigmapsf": pdf_valid[["csigmapsf", "is_new"]].apply(
                    lambda x: x["csigmapsf"][x["is_new"]], axis=1
                ),
                "cfid": pdf_valid[["cfid", "is_new"]].apply(
                    lambda x: x["cfid"][x["is_new"]], axis=1
                ),
            }
        )

        # Add it to the history alerts
        for field in ["cjd", "cmagpsf", "csigmapsf", "cfid"]:
            lcs[field] = np.array(
                lcs[field].apply(list) + current_night[field].apply(list)
            )

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
            extracted = np.sum(features.isna(), axis=1) == 0
            probas[extracted] = clf.predict_proba(
                features.loc[extracted, clf.feature_names_in_]
            )[:, 1]

            probas_total[mask_valid] = probas
            return pd.Series(probas_total)

        else:
            return pd.Series([-1.0] * len(pdf))


def get_and_format(ZTF_name):
    """Use the fink API to collect the full light curve sources using ZTF names.

    Parameters
    ----------
    ZTF_name: list
        List of objectId (ZTF names)

    Returns
    -------
    pd.DataFrame
        DataFrame containing all light curve information.
        1 row = 1 source. Returns None if the list is empty.

    Example
    -------
    >>> get_and_format([]) is None
    True
    >>> get_and_format(["toto"]) is None
    True
    >>> get_and_format("toto")
    Traceback (most recent call last):
    ...
    TypeError: ZTF_name should be a list of str
    >>> data = get_and_format(["ZTF21abfmbix", "ZTF21abfmbix"])
    >>> (data["distnr"].iloc[0] > 3.0) & (data["distnr"].iloc[0] < 3.5)
    True
    >>> list(data.columns) == ['objectId', 'cjd', 'cmagpsf', 'csigmapsf', 'cfid', 'distnr']
    True
    >>> len(data['cjd'].iloc[0]) >= 14
    True
    """
    if type(ZTF_name) is not list:
        raise TypeError("ZTF_name should be a list of str")

    if len(ZTF_name) == 0:
        return None

    # Initialize an empty list to hold each DataFrame
    dataframes = []

    for _id, name in enumerate(ZTF_name):
        r = requests.post(
            "https://api.fink-portal.org/api/v1/objects",
            json={
                "objectId": name,
                "columns": "i:objectId,i:jd,i:magpsf,i:sigmapsf,i:fid,i:jd,i:distnr,d:tag",
                "output-format": "json",
                "withupperlim": "True",
            },
        )

        # Format output in a DataFrame
        pdf = pd.read_json(io.BytesIO(r.content))

        if len(pdf) != 0:
            pdf["id"] = _id
            pdf = pdf.sort_values("i:jd")
            valid = (pdf["d:tag"] == "valid") | (pdf["d:tag"] == "badquality")
            pdf = pdf[valid]

            if not pdf.empty:
                # Append valid DataFrame to the list
                dataframes.append(pdf)

    if dataframes:
        combined_pdf = pd.concat(dataframes)
        pdfs = [group for _, group in combined_pdf.groupby("id")]
        lcs = pd.DataFrame(
            data={
                "objectId": [lc["i:objectId"].iloc[0] for lc in pdfs],
                "cjd": [np.array(lc["i:jd"].values, dtype=float) for lc in pdfs],
                "cmagpsf": [
                    np.array(lc["i:magpsf"].values, dtype=float) for lc in pdfs
                ],
                "csigmapsf": [
                    np.array(lc["i:sigmapsf"].values, dtype=float) for lc in pdfs
                ],
                "cfid": [np.array(lc["i:fid"].values, dtype=int) for lc in pdfs],
                "distnr": [np.mean(lc["i:distnr"]) for lc in pdfs],
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
