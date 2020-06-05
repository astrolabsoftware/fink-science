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

from fink_science.microlensing.classifier import load_external_model
from fink_science.microlensing.classifier import load_mulens_schema_twobands
from fink_science.conversion import dc_mag

from LIA import microlensing_classifier

from fink_science.tester import spark_unit_tests

def mulens(
        fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos, model_path=None):
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
    model_path: Spark DataFrame Column, optional
        Path to the trained models. Default is None, in which case the default
        models `data/models/rf.sav` and `data/models/pca.sav` are loaded.

    Returns
    ----------
    out: list
        Returns the class (string) and microlensing score (double) ordered as
        [class_band_1, ml_score_band1, class_band_2, ml_score_band2]

    Examples
    ---------
    >>> from fink_science.utilities import concat_col
    >>> from pyspark.sql import functions as F

    >>> df = spark.read.load(ztf_alert_sample)

    >>> schema = load_mulens_schema_twobands()

    >>> t = udf(mulens, schema)

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

    # Load pre-trained model `rf` and `pca`
    if model_path is not None:
        rf, pca = load_external_model(model_path.values[0])
    else:
        curdir = os.path.dirname(os.path.abspath(__file__))
        model_path = curdir + '/../data/models/'
        rf, pca = load_external_model(model_path)

    # Select only valid measurements (not upper limits)
    maskNotNone = np.array(magpsf) != None

    out = []
    for filt in [1, 2]:
        maskFilter = np.array(fid) == filt
        m = maskNotNone * maskFilter

        # Reject if less than 6 measurements
        if np.sum(m) < 6:
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
    ztf_alert_sample = 'fink_science/data/alerts/alerts.parquet'
    globs["ztf_alert_sample"] = ztf_alert_sample

    model_path = 'fink_science/data/models'
    globs["model_path"] = model_path

    # Run the test suite
    spark_unit_tests(globs)
