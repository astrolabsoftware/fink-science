# Copyright 2020 AstroLab Software
# Author: Julien Peloton
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
from pyspark.sql.functions import udf, col

import numpy as np

import os
import warnings

from fink_science import __file__
from fink_science.microlensing.classifier import load_external_model
from fink_science.microlensing.classifier import load_mulens_schema_twobands
from fink_science.conversion import dc_mag

from LIA import microlensing_classifier

from fink_science.tester import spark_unit_tests

def mulens(
        fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos, rf, pca):
    """ Returns the predicted class (among microlensing, variable star,
    cataclysmic event, and constant event) & probability of an alert to be
    a microlensing event in each band using a Random Forest Classifier.

    Parameters
    ----------
    fid: Spark DataFrame Column
        Filter IDs (int)
    magpsf, sigmapsf: Spark DataFrame Columns
        Magnitude from PSF-fit photometry, and 1-sigma error
    magnr, sigmagnr: Spark DataFrame Columns
        Magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
    magzpsci: Spark DataFrame Column
        Magnitude zero point for photometry estimates
    isdiffpos: Spark DataFrame Column
        t => candidate is from positive (sci minus ref) subtraction
        f => candidate is from negative (ref minus sci) subtraction
    rf: RandomForestClassifier
        sklearn.ensemble._forest.RandomForestClassifier
    pca: PCA
        sklearn.decomposition._pca.PCA

    Returns
    ----------
    out: list
        Returns the class (string) and microlensing score (double) ordered as
        [class_band_1, ml_score_band1, class_band_2, ml_score_band2]

    Examples
    ---------
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    # wrapper to pass broadcasted values
    >>> def mulens_wrapper(fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos):
    ...     return mulens(fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, rfbcast.value, pcabcast.value)

    >>> df = spark.read.load(ztf_alert_sample)

    >>> schema = load_mulens_schema_twobands()

    # Required alert columns
    >>> what = [
    ...    'fid', 'magpsf', 'sigmapsf',
    ...    'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

    # Use for creating temp name
    >>> prefix = 'c'
    >>> what_prefix = [prefix + i for i in what]

    # Append temp columns with historical + current measurements
    >>> for colname in what:
    ...    df = concat_col(df, colname, prefix=prefix)

    >>> rf, pca = load_external_model(model_path)
    >>> rfbcast = spark.sparkContext.broadcast(rf)
    >>> pcabcast = spark.sparkContext.broadcast(pca)

    >>> t = udf(mulens_wrapper, schema)
    >>> args = [col(i) for i in what_prefix]
    >>> df_mulens = df.withColumn('mulens', t(*args))

    # Drop temp columns
    >>> df_mulens = df_mulens.drop(*what_prefix)

    >>> df_mulens.agg({"mulens.ml_score_1": "min"}).collect()[0][0]
    0.0

    >>> df_mulens.agg({"mulens.ml_score_1": "max"}).collect()[0][0] < 1.0
    True
    """
    warnings.filterwarnings('ignore')

    # Select only valid measurements (not upper limits)
    maskNotNone = np.array(magpsf) != None

    out = []
    for filt in [1, 2]:
        maskFilter = np.array(fid) == filt
        m = maskNotNone * maskFilter

        # Reject if less than 10 measurements
        if np.sum(m) < 10:
            out.extend(['', 0.0])
            continue

        # Compute DC mag
        mag, err = np.array([
            dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
            for i in zip(
                np.array(fid)[m],
                np.array(magpsf)[m],
                np.array(sigmapsf)[m],
                np.array(magnr)[m],
                np.array(sigmagnr)[m],
                np.array(magzpsci)[m],
                np.array(isdiffpos)[m])
        ]).T

        # Run the classifier
        output = microlensing_classifier.predict(mag, err, rf, pca)

        # Update the results
        out.extend([str(output[0]), float(output[1][0])])

    return out


if __name__ == "__main__":
    """ Execute the test suite """

    globs = globals()
    path = os.path.dirname(__file__)
    ztf_alert_sample = 'file://{}/data/alerts/alerts.parquet'.format(path)
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path = '{}/data/models'.format(path)
    globs["model_path"] = model_path

    # Run the test suite
    spark_unit_tests(globs)
